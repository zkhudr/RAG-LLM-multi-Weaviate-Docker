import re
import logging
import sys
import json
import argparse
import numpy as np
import spacy
import weaviate



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter # To count keyword frequency
from pathlib import Path
from config import cfg # Import the central config object
from weaviate.exceptions import WeaviateConnectionError
from weaviate.classes.config import DataType
# --- Configuration & Parameters ---
DEFAULT_COLLECTION_NAME = "Industrial_tech" # Keep this or read from cfg later?
DEFAULT_KEYBERT_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_SPACY_MODEL = 'en_core_web_sm'
DEFAULT_TOP_N_PER_DOC = 20 # Keywords to extract *per document*
DEFAULT_N_GRAM_RANGE = (1, 3) # Reduced N-gram range slightly to limit candidates
DEFAULT_MIN_KEYWORD_LENGTH = 4
DEFAULT_EXTRACTION_DIVERSITY = 0.1 # MMR diversity for KeyBERT
DEFAULT_POS_TAGS = {"NOUN", "PROPN"} # Focused on Nouns/Proper Nouns
DEFAULT_FINAL_TOP_N = 200 # Number of keywords after aggregation
DEFAULT_MIN_DOC_FREQ = 2 # Minimum documents a keyword must appear in
DEFAULT_N_CLUSTERS = 4 # Number of clusters for KMeans


# --- Logging Setup ---
def setup_logging(force_utf8=False, level=logging.INFO):
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if force_utf8:
        try:
            stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except Exception:
            pass
    logging.basicConfig(handlers=[stream_handler], level=level, force=True)

setup_logging(force_utf8=True)

import logging
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Keyword Extraction Pipeline (Per-Document KeyBERT).")

    # ─── Legacy alias: accept --min_doc_freq and treat it as absolute min-doc-frequency ─────────────────────────────────────
    parser.add_argument(
        "--min_doc_freq",
        type=int,
        default=None,
        help="(Legacy) shorthand for --min_doc_freq_abs"
    )

    # ─── Core extraction flags ────────────────────────────────────────────────────────────────────────────────────────
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Weaviate collection name")
    parser.add_argument("--keybert_model", default=DEFAULT_KEYBERT_MODEL, help="Sentence Transformer model for KeyBERT")
    parser.add_argument("--spacy_model", default=DEFAULT_SPACY_MODEL, help="SpaCy model for POS tagging")
    parser.add_argument("--top_n_per_doc", type=int, default=DEFAULT_TOP_N_PER_DOC, help="Keywords to extract per document with KeyBERT")
    parser.add_argument("--final_top_n", type=int, default=DEFAULT_FINAL_TOP_N, help="Final number of keywords after aggregation and filtering")
    parser.add_argument("--ngram_range", type=lambda s: tuple(map(int, s.split(','))), default=DEFAULT_N_GRAM_RANGE, help="N-gram range for keywords (e.g., '1,2')")
    parser.add_argument("--min_len", type=int, default=DEFAULT_MIN_KEYWORD_LENGTH, help="Minimum character length for final keywords")
    parser.add_argument("--extraction_diversity", type=float, default=DEFAULT_EXTRACTION_DIVERSITY, help="Diversity setting (0-1) for KeyBERT's MMR")
    parser.add_argument("--pos_tags", type=lambda s: set(s.split(',')), default=DEFAULT_POS_TAGS, help="Comma-separated POS tags to keep (e.g., 'NOUN,PROPN')")
    parser.add_argument("--min_doc_freq_abs", type=int, default=None, help="Absolute min-doc-frequency (overrides fraction if set)")
    parser.add_argument("--min_doc_freq_frac", type=float, default=None, help="Fractional min-doc-frequency (0.0–1.0). Rounded up * total_docs")
    parser.add_argument("--no_pos_filter", action="store_true", help="Disable POS tag filtering")

    # ─── Legacy/UI-only aliases to absorb extra flags from Flask/UI ─────────────────────────────────────────────────
    parser.add_argument(
        "--diversity",
        type=float,
        default=None,
        help="(Alias) same as --extraction_diversity"
    )
    parser.add_argument(
        "--docFreqMode",
        choices=["absolute","fraction"],
        default=None,
        help="(Ignored) UI flag to pick abs vs frac input"
    )
    parser.add_argument(
        "--ingestion.MIN_QUALITY_SCORE",
        type=float,
        default=None,
        help="(Ignored) override for ingestion.MIN_QUALITY_SCORE"
    )

    args = parser.parse_args()

    # ─── Map legacy shorthand to real param ─────────────────────────────────────────────────────────────────────────
    if args.min_doc_freq is not None:
        if args.min_doc_freq_abs is None and args.min_doc_freq_frac is None:
            args.min_doc_freq_abs = args.min_doc_freq

    # ─── Map UI alias into real extraction flag ─────────────────────────────────────────────────────────────────────
    if args.diversity is not None:
        args.extraction_diversity = args.diversity

    # ─── ngram_range sanity check ─────────────────────────────────────────────────────────────────────────────────
    if len(args.ngram_range) != 2 or args.ngram_range[0] < 1 or args.ngram_range[1] < args.ngram_range[0]:
        parser.error("Invalid ngram_range. Use format like '1,2' or '1,3'.")

    return args



def load_spacy_model(model_name: str):
    """Loads spaCy model."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.warning(f"SpaCy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    logger.info(f"Loaded spaCy model '{model_name}'.")
    return nlp

# Assuming the config is loaded using Pydantic model

def get_all_content_from_weaviate(client: weaviate.WeaviateClient, collection_name: str) -> list[str]:
    """Retrieves non-empty text content from all objects using available text properties (Weaviate v4 syntax)."""
    texts = []
    try:
        logger.info(f"Attempting to access collection '{collection_name}'...")
        # Check if collection exists using v4 method
        if not client.collections.exists(collection_name):
             logger.error(f"Collection '{collection_name}' does not exist in Weaviate.")
             # Optionally, list available collections if helpful
             all_collections = client.collections.list_all(simple=True) # Get simple list
             logger.info(f"Available collections: {[c.name for c in all_collections]}")
             return []

        # Get the collection object (v4 style)
        collection = client.collections.get(collection_name)
        logger.info(f"Successfully accessed collection '{collection_name}'.")

        # Get properties from the collection config (v4 style)
        config = collection.config.get()
        text_properties = []
        for prop in config.properties:
            # Check for TEXT or TEXT_ARRAY data types USING the imported DataType [2][6]
            if prop.data_type == DataType.TEXT or prop.data_type == DataType.TEXT_ARRAY:
                 text_properties.append(prop.name)

        if not text_properties:
            # Fallback if no text properties are explicitly found (less reliable)
            text_properties = ["content", "text", "body", "description", "source"] # Keep common fallbacks
            logger.warning(f"No explicit TEXT properties found in schema for '{collection_name}'. Trying common properties: {text_properties}")
        else:
            logger.info(f"Identified text properties to check: {text_properties}")

        logger.info(f"Iterating through objects in '{collection_name}' to retrieve content from properties: {text_properties}...")

        object_count = 0
        processed_count = 0
        # Iterate through objects using the v4 iterator [7][4]
        # Fetch only the identified text properties
        fetch_props = list(set(text_properties)) # Ensure unique props

        for obj in collection.iterator(return_properties=fetch_props):
            processed_count += 1
            found_content_in_obj = False
            # Access properties via obj.properties dictionary
            properties = obj.properties

            for prop_name in text_properties:
                content = properties.get(prop_name)
                if content:
                    # Handle both string and list of strings (TEXT_ARRAY)
                    if isinstance(content, list): # Handle TEXT_ARRAY
                        for item in content:
                             if item and isinstance(item, str) and item.strip():
                                 texts.append(item.strip())
                                 found_content_in_obj = True
                    elif isinstance(content, str) and content.strip(): # Handle TEXT
                        texts.append(content.strip())
                        found_content_in_obj = True

            if found_content_in_obj:
                object_count += 1 # Count objects from which we got *any* text

            if processed_count % 500 == 0: # Log progress periodically
                 logger.info(f"Processed {processed_count} objects...")

        logger.info(f"Finished iterating. Retrieved {len(texts)} non-empty text entries from {object_count} objects (processed {processed_count} total objects).")

        if not texts:
            # Check total object count in collection for context
            total_obj_count = collection.aggregate.over_all(total_count=True).total_count
            logger.warning(f"Failed to extract any text content from '{collection_name}'. The collection has {total_obj_count} objects.")
            if total_obj_count > 0:
                 logger.warning(f"Ensure the specified text properties {text_properties} contain data in objects.")

        return texts

    except WeaviateConnectionError as conn_e:
         logger.critical(f"Weaviate connection error while querying '{collection_name}': {conn_e}", exc_info=True)
         raise # Re-raise or handle as needed
    except Exception as e:
        # Log the specific error including the type
        logger.error(f"Failed to retrieve content from Weaviate collection '{collection_name}' ({type(e).__name__}): {e}", exc_info=True)
        return [] # Return empty list on other errors



# <<< MODIFIED: Run KeyBERT per document >>>
def run_keybert_per_document(texts: list[str], model_name: str, top_n_per_doc: int, ngram_range: tuple[int, int], extraction_diversity: float) -> dict[str, float]:
    """
    Extracts keywords using KeyBERT for each document individually and aggregates scores.
    Returns a dictionary of {keyword: max_score}.
    """
    if not texts:
        logger.warning("No texts provided for KeyBERT extraction.")
        return {}

    aggregated_keywords = {} # Store {keyword: max_score}
    keyword_doc_frequency = Counter() # Store {keyword: doc_count}

    try:
        logger.info(f"Initializing KeyBERT with model '{model_name}'...")
        transformer_model = SentenceTransformer(model_name)
        kw_model = KeyBERT(model=transformer_model)
        logger.info("KeyBERT model initialized.")

        logger.info(f"Running KeyBERT extraction per document (top_n={top_n_per_doc}, ngram_range={ngram_range}, diversity={extraction_diversity})...")
        processed_docs = 0
        for i, text in enumerate(texts):
            print(f"🔄 Running KeyBERT on doc {i+1}/{len(texts)}")
            try:
                # Limit text length if necessary to avoid individual doc issues
                # max_len = 50000 # Example limit
                # truncated_text = text[:max_len]

                keywords_with_scores = kw_model.extract_keywords(
                    text, # Process individual text
                    keyphrase_ngram_range=ngram_range,
                    stop_words='english',
                    use_mmr=True,
                    diversity=extraction_diversity,
                    top_n=top_n_per_doc # Extract top N for THIS doc
                )

                if keywords_with_scores:
                    # Aggregate results
                    unique_keywords_in_doc = set()
                    for kw, score in keywords_with_scores:
                        kw_lower = kw.lower() # Normalize keyword case
                        unique_keywords_in_doc.add(kw_lower)
                        # Store the highest score seen for this keyword across all docs
                        aggregated_keywords[kw_lower] = max(aggregated_keywords.get(kw_lower, 0.0), float(score))
                    # Increment document frequency count for keywords found in this doc
                    keyword_doc_frequency.update(unique_keywords_in_doc)

                processed_docs += 1
                if processed_docs % 100 == 0:
                    logger.info(f"Processed {processed_docs}/{len(texts)} documents...")

            except Exception as doc_e:
                logger.error(f"Error processing document {i} with KeyBERT: {doc_e}")
                continue # Skip to next document

        logger.info(f"KeyBERT extraction finished. Aggregated {len(aggregated_keywords)} unique keywords.")
        # Return both aggregated scores and document frequencies
        return aggregated_keywords, keyword_doc_frequency

    except Exception as e:
        logger.error(f"Error during KeyBERT initialization or processing: {e}", exc_info=True)
        return {}, Counter() # Return empty results

# <<< MODIFIED: Filter based on aggregated results >>>
def apply_final_filters(
    aggregated_keywords: dict[str, float],
    keyword_doc_frequency: Counter,
    nlp: spacy.language.Language,
    valid_pos_tags: set[str],
    min_len: int,
    min_doc_freq: int,
    apply_pos_filter: bool = True
) -> list[tuple[str, float]]:
    """
    Filters aggregated keywords based on doc frequency, POS, length, patterns.
    Returns sorted list: [(keyword, max_score)].
    """
    if not aggregated_keywords:
        return []

    term_scores_initial = list(aggregated_keywords.items())
    # Sort by score descending before filtering
    term_scores_initial.sort(key=lambda x: x[1], reverse=True)

    filtered_scores = []
    logger.info(f"Filtering {len(term_scores_initial)} aggregated keywords (min_len={min_len}, min_doc_freq={min_doc_freq}, POS filter: {apply_pos_filter})...")
    initial_count = len(term_scores_initial)
    removed_freq = 0
    removed_pos = 0
    removed_len = 0
    removed_pattern = 0

    # --- Pre-compile Regex Patterns (same as before) ---
    patterns_to_remove = [
        re.compile(r'^\d+$'), re.compile(r'^cid\s*\d+$'), re.compile(r'^overview\s+\d+$'),
        re.compile(r'^\d+\s+\d+$'), re.compile(r'^\d+m[ms]?$'),
        re.compile(r'^\d+(?:mm|cc|vdc|vac|psi|kg|lb|hz|khz|mhz|kv|kva|kw|kva)$'),
        re.compile(r'^[a-zA-Z]*\d+[a-zA-Z\d]*$'), re.compile(r'^_+.*|.*_+$'),
        re.compile(r'[<>#@*(){}\[\]\\]'),
        re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', re.IGNORECASE),
    ]
    # ---------------------------------

    for term, score in term_scores_initial:

        # --- Min Document Frequency Filter ---
        if keyword_doc_frequency.get(term, 0) < min_doc_freq:
            removed_freq += 1
            continue

        # --- Length Filter ---
        if len(term) < min_len:
            removed_len += 1
            continue

        # --- Pattern Filtering ---
        is_pattern_match = False
        for pattern in patterns_to_remove:
            if pattern.search(term): # Match against original case term too? Use term.lower()?
                is_pattern_match = True
                removed_pattern += 1
                break
        if is_pattern_match:
            continue

        # --- POS Tag Filtering ---
        if apply_pos_filter:
            doc = nlp(term)
            is_valid_pos = False
            if len(doc) > 0:
                 if all(token.is_punct or token.is_space or token.is_stop for token in doc):
                      removed_pos +=1
                      continue
                 for token in doc:
                      if token.pos_ in valid_pos_tags and not token.is_stop:
                           is_valid_pos = True
                           break
            if not is_valid_pos:
                removed_pos += 1
                continue

        # If passed all filters
        filtered_scores.append((term, score))

    removed_count = initial_count - len(filtered_scores)
    logger.info(f"Filtering complete. Removed {removed_count} keywords ({removed_freq} freq, {removed_len} len, {removed_pattern} pattern, {removed_pos} POS). Returning {len(filtered_scores)} keywords.")
    # Final list is already sorted by score
    return filtered_scores

# --- Cluster Keywords ---
def cluster_keywords(keywords: list[str], n_clusters: int):
    """Clusters keywords using KMeans."""
    # ... (Same as before) ...
    if not keywords:
        logger.warning("No keywords provided for clustering.")
        return {}
    if len(keywords) < n_clusters:
         logger.warning(f"Number of keywords ({len(keywords)}) is less than n_clusters ({n_clusters}). Reducing n_clusters.")
         n_clusters = max(1, len(keywords))

    try:
        logger.info(f"Clustering {len(keywords)} keywords into {n_clusters} themes...")
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(keywords)

        if X.shape[1] == 0:
            logger.error("Vocabulary is empty after vectorizing keywords for clustering.")
            return {}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(X)
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(keywords[idx])
        logger.info("Clustering complete.")
        return clusters

    except Exception as e:
        logger.error(f"Error during keyword clustering: {e}", exc_info=True)
        return {}
    

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()
    client = None
    nlp_model = None
    try:
        # 1) Load spaCy model
        nlp_model = load_spacy_model(args.spacy_model)

        # 2) Initialize Weaviate client via your central factory
        from ingest_docs_v7 import PipelineConfig
        client = PipelineConfig.get_client()
        if client is None or not client.is_live():
            logger.critical("Fatal: Could not obtain a live Weaviate client.")
            sys.exit(1)

        # 3) Fetch all texts from Weaviate
        all_texts = get_all_content_from_weaviate(client, args.collection)

        # 4) Run KeyBERT → aggregation → filtering → clustering
        if all_texts:
            aggregated, freqs = run_keybert_per_document(
                all_texts,
                args.keybert_model,
                args.top_n_per_doc,
                args.ngram_range,
                args.extraction_diversity
            )
            final_keywords = apply_final_filters(
                aggregated, freqs, nlp_model,
                set(args.pos_tags), args.min_len, 
                args.min_doc_freq_abs or math.ceil(args.min_doc_freq_frac * len(all_texts)),
                apply_pos_filter=not args.no_pos_filter
            )
            # (… your existing print(), file-write, cluster_keywords() code …)
        else:
            print("No content retrieved from Weaviate. Cannot perform keyword extraction.")

    except Exception as e:
        logger.critical(f"Unexpected error in keyword builder: {e}", exc_info=True)
        print(f" [ ERROR] Keyword Builder failed: {e}")
        sys.exit(1)

    finally:
        # 5) Always close the Weaviate client
        if client:
            try:
                if client.is_live():
                    client.close()
                    logger.info("Weaviate client closed after keyword builder run.")
                else:
                    client.close()
                    logger.warning("Closed Weaviate client (was not live).")
            except Exception as close_err:
                logger.error(f"Error closing Weaviate client: {close_err}", exc_info=True)

                 
    # 1. Load ingested_docs.json
    docs_file = Path(cfg.paths.DOCUMENT_DIR) / "ingested_docs.json"
    if docs_file.exists():
        with open(docs_file, "r", encoding="utf-8") as f:
            all_files = json.load(f)
        total_docs = len(all_files)
    else:
        # fallback to whatever is actually loaded from Weaviate
        total_docs = len(all_texts) if 'all_texts' in locals() else 0
    
    # 2. Decide which threshold to use
    if args.min_doc_freq_abs is not None:
        min_doc_freq = args.min_doc_freq_abs
        mode = "absolute"
    elif args.min_doc_freq_frac is not None:
        min_doc_freq = math.ceil(args.min_doc_freq_frac * total_docs)
        mode = "fraction"
    else:
        # fallback to old arg
        min_doc_freq = args.min_doc_freq
        mode = "absolute"
    
    # 3. Log both forms
    computed_frac = min_doc_freq / total_docs if total_docs > 0 else 0
    logger.info(
        f"Using min_doc_freq={min_doc_freq} "
        f"({computed_frac:.2%} of {total_docs} docs) "
        f"mode={mode}"
    )
    # ====== END OF ADDED CODE ======
    
    client = None
    nlp_model = None

    if not cfg or not cfg.retrieval or not cfg.model:
        logger.critical("CRITICAL: Central configuration (cfg) not loaded or incomplete. Cannot proceed.")
        sys.exit(1)

    try:
        # Load spaCy model once
        nlp_model = load_spacy_model(args.spacy_model)

        # --- Connect to Weaviate using cfg ---
        w_host = cfg.retrieval.WEAVIATE_HOST
        w_http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        w_grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
        # Ensure w_timeout_tuple is a tuple e.g., (10, 120) from config
        w_timeout_tuple = cfg.retrieval.WEAVIATE_TIMEOUT
        w_collection_name = args.collection # Use collection name from args/defaults

        logger.info(f"Connecting to Weaviate at {w_host}:{w_http_port} (gRPC: {w_grpc_port}) using central config...")
        logger.info(f"Using connection timeout: {w_timeout_tuple}")

        # CORRECTED CONNECTION using connect_to_local and AdditionalConfig
        try:
            # Create AdditionalConfig object specifically for the timeout
            # The timeout parameter within AdditionalConfig expects the tuple[2]
            additional_config = weaviate.config.AdditionalConfig(timeout=w_timeout_tuple)

            client = weaviate.connect_to_local(
                host=w_host,
                port=w_http_port,
                grpc_port=w_grpc_port,
                # headers=... # Add API keys or auth headers if needed via headers= parameter
                additional_config=additional_config # Pass the config object here
            )
            # connect_to_local implicitly tries to connect.
            # Explicit connect() might not be needed but calling is_ready() is crucial.

            if not client.is_ready(): # Check readiness using is_ready() for v4[2][5]
                 raise WeaviateConnectionError("Weaviate client connected but server reports not ready.")

            logger.info(f"Successfully connected to Weaviate at {w_host}:{w_http_port} and server is ready.")

        except TypeError as te:
            # Catch the specific error if it persists, helps debugging
            logger.critical(f"Fatal error: Connection parameter issue. Details: {te}\n"
                            f"Host: {w_host}, HTTP: {w_http_port}, gRPC: {w_grpc_port}, Timeout: {w_timeout_tuple}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Fatal error: Connection to Weaviate failed. Details: {e}\n"
                            f"Host: {w_host}, HTTP: {w_http_port}, gRPC: {w_grpc_port}, Timeout: {w_timeout_tuple}", exc_info=True)
            sys.exit(1)

        # --- Get content using the CORRECTED function from previous step ---
        all_texts = get_all_content_from_weaviate(client, w_collection_name)

        # Debugging: Check if texts were retrieved
        # print(f"Retrieved {len(all_texts)} text items.")
        # if all_texts:
        #    print(f"First item preview: {repr(all_texts[0][:100])}...")

        # Extract Keywords using KeyBERT (Per Document + Aggregation)
        if all_texts:
            logger.info(f"Starting KeyBERT extraction on {len(all_texts)} documents...") # Added log
            aggregated_keywords_scores, keyword_frequencies = run_keybert_per_document(
                all_texts,
                args.keybert_model,
                args.top_n_per_doc,
                args.ngram_range,
                args.extraction_diversity
            )

            # Filter Keywords (Doc Freq, Linguistic + Pattern)
            if aggregated_keywords_scores:
                logger.info("Applying final filters to aggregated keywords...") # Added log
                final_keywords_with_scores = apply_final_filters(
                        aggregated_keywords_scores,
                        keyword_frequencies,
                        nlp_model,
                        args.pos_tags,
                        args.min_len,
                        min_doc_freq,  # Using our calculated threshold
                        apply_pos_filter=not args.no_pos_filter
                )

                # Select final Top N from filtered list
                final_keywords_with_scores = final_keywords_with_scores[:args.final_top_n]

                if final_keywords_with_scores:
                     # Print Top N Final Keywords
                    print(f"\n--- Top {len(final_keywords_with_scores)} Filtered Domain Keywords (ranked by Max KeyBERT score) ---")
                    for term, score in final_keywords_with_scores:
                        print(f"{term} : {score:.4f}")
                    print("-----------------------------------------------------------------------------")

                    # Prepare keywords for file writing
                    keyword_list_for_file = [term for term, _ in final_keywords_with_scores]
                    output_filename = "auto_domain_keywords.txt"
                    try:
                        logger.info(f"Writing {len(keyword_list_for_file)} keywords to {output_filename}...")
                        with open(output_filename, "w", encoding="utf-8") as f:
                            f.write(", ".join(keyword_list_for_file))
                        logger.info(f"Successfully wrote keywords to {output_filename}")
                    except IOError as e:
                         logger.error(f"Failed to write keywords to {output_filename}: {e}")

                    # Cluster the final keywords
                    keyword_list_for_clustering = [term for term, _ in final_keywords_with_scores]
                    num_clusters = getattr(args, 'n_clusters', DEFAULT_N_CLUSTERS) # Use default or add arg
                    logger.info(f"Starting keyword clustering into {num_clusters} clusters...") # Added log
                    clustered_keywords = cluster_keywords(keyword_list_for_clustering, n_clusters=num_clusters)

                    if clustered_keywords:
                        print("\n--- Keyword Clusters (Themes) ---")
                        for cluster_id, keywords_in_cluster in clustered_keywords.items():
                            print(f"Cluster {cluster_id + 1}: {', '.join(keywords_in_cluster[:15])}" + ("..." if len(keywords_in_cluster) > 15 else ""))
                        print("---------------------------------------")
                    else:
                        logger.warning("Clustering did not produce results.")

                else:
                    print("No keywords remained after all filtering steps.")
            else:
                print("KeyBERT did not extract any keywords after aggregation.")
        else:
            print("No content retrieved from Weaviate. Cannot perform keyword extraction.") # Modified message

    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main execution block: {str(e)}", exc_info=True)
        print(f" Failure: {str(e)}")
        sys.exit(1)

    finally:
        # Use is_connected() for v4 to check before closing
        if client and client.is_connected():
            logger.info("Closing Weaviate client connection.")
            client.close()
        elif client:
             logger.warning("Client object exists but was not connected.")