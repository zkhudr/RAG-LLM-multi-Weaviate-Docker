import re
import logging
import sys
import json
import argparse
from collections import Counter # To count keyword frequency

# NLP & ML Libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Weaviate v4 imports
import weaviate
from weaviate.exceptions import WeaviateConnectionError

# --- Configuration & Parameters ---
DEFAULT_WEAVIATE_HOST = "localhost"
DEFAULT_WEAVIATE_HTTP_PORT = 8080
DEFAULT_WEAVIATE_GRPC_PORT = 50051
DEFAULT_COLLECTION_NAME = "industrial_tech"
DEFAULT_KEYBERT_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_SPACY_MODEL = 'en_core_web_sm'
DEFAULT_TOP_N_PER_DOC = 10 # Keywords to extract *per document*
DEFAULT_N_GRAM_RANGE = (1, 2) # Reduced N-gram range slightly to limit candidates
DEFAULT_MIN_KEYWORD_LENGTH = 4
DEFAULT_DIVERSITY = 0.7 # MMR diversity for KeyBERT
DEFAULT_POS_TAGS = {"NOUN", "PROPN"} # Focused on Nouns/Proper Nouns
DEFAULT_FINAL_TOP_N = 100 # Number of keywords after aggregation
DEFAULT_MIN_DOC_FREQ = 2 # Minimum documents a keyword must appear in
DEFAULT_N_CLUSTERS = 7 # Number of clusters for KMeans


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Keyword Extraction Pipeline (Per-Document KeyBERT).")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Weaviate collection name")
    parser.add_argument("--keybert_model", default=DEFAULT_KEYBERT_MODEL, help="Sentence Transformer model for KeyBERT")
    parser.add_argument("--spacy_model", default=DEFAULT_SPACY_MODEL, help="SpaCy model for POS tagging")
    parser.add_argument("--top_n_per_doc", type=int, default=DEFAULT_TOP_N_PER_DOC, help="Keywords to extract per document with KeyBERT")
    parser.add_argument("--final_top_n", type=int, default=DEFAULT_FINAL_TOP_N, help="Final number of keywords after aggregation and filtering")
    parser.add_argument("--ngram_range", type=lambda s: tuple(map(int, s.split(','))), default=DEFAULT_N_GRAM_RANGE, help="N-gram range for keywords (e.g., '1,2')")
    parser.add_argument("--min_len", type=int, default=DEFAULT_MIN_KEYWORD_LENGTH, help="Minimum character length for final keywords")
    parser.add_argument("--diversity", type=float, default=DEFAULT_DIVERSITY, help="Diversity setting (0-1) for KeyBERT's MMR")
    parser.add_argument("--pos_tags", type=lambda s: set(s.split(',')), default=DEFAULT_POS_TAGS, help="Comma-separated POS tags to keep (e.g., 'NOUN,PROPN')")
    parser.add_argument("--min_doc_freq", type=int, default=DEFAULT_MIN_DOC_FREQ, help="Minimum number of documents a keyword must appear in")
    parser.add_argument("--no_pos_filter", action='store_true', help="Disable POS tag filtering")

    args = parser.parse_args()
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

def get_all_content_from_weaviate(client: weaviate.Client, collection_name: str) -> list[str]:
    """Retrieves non-empty 'content' from all objects."""
    # ... (Same as before) ...
    texts = []
    try:
        collection = client.collections.get(collection_name)
        logger.info(f"Iterating through all objects in '{collection_name}' to retrieve content...")
        object_count = 0
        for obj in collection.iterator(return_properties=["content"], include_vector=False):
            object_count += 1
            content = obj.properties.get("content")
            if content and isinstance(content, str) and content.strip(): # Ensure non-empty string
                texts.append(content)
            # else:
            #     logger.debug(f"Object UUID {obj.uuid} missing 'content' property or it's empty/not a string.")

        logger.info(f"Retrieved non-empty content from {len(texts)} objects (processed {object_count} total objects).")
        if object_count > 0 and len(texts) == 0:
             logger.error("Processed objects but failed to extract any valid text content!")
        return texts
    except Exception as e:
        logger.error(f"Failed to retrieve content from Weaviate collection '{collection_name}': {e}", exc_info=True)
        return []

# <<< MODIFIED: Run KeyBERT per document >>>
def run_keybert_per_document(texts: list[str], model_name: str, top_n_per_doc: int, ngram_range: tuple[int, int], diversity: float) -> dict[str, float]:
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

        logger.info(f"Running KeyBERT extraction per document (top_n={top_n_per_doc}, ngram_range={ngram_range}, diversity={diversity})...")
        processed_docs = 0
        for i, text in enumerate(texts):
            try:
                # Limit text length if necessary to avoid individual doc issues
                # max_len = 50000 # Example limit
                # truncated_text = text[:max_len]

                keywords_with_scores = kw_model.extract_keywords(
                    text, # Process individual text
                    keyphrase_ngram_range=ngram_range,
                    stop_words='english',
                    use_mmr=True,
                    diversity=diversity,
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

# --- Cluster Keywords (Remains the same) ---
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
        # Load spaCy model once
        nlp_model = load_spacy_model(args.spacy_model)

        # Connect to Weaviate
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_local(
            host=DEFAULT_WEAVIATE_HOST,
            port=DEFAULT_WEAVIATE_HTTP_PORT,
            grpc_port=DEFAULT_WEAVIATE_GRPC_PORT
        )
        if not client.is_ready():
            raise WeaviateConnectionError("Client connected but not ready.")
        logger.info("Weaviate connection successful.")

        # Get content
        all_texts = get_all_content_from_weaviate(client, args.collection)

        # Extract Keywords using KeyBERT (Per Document + Aggregation)
        if all_texts:
            # Returns {keyword: max_score}, {keyword: doc_freq}
            aggregated_keywords_scores, keyword_frequencies = run_keybert_per_document(
                all_texts,
                args.keybert_model,
                args.top_n_per_doc,
                args.ngram_range,
                args.diversity
            )

            # Filter Keywords (Doc Freq, Linguistic + Pattern)
            if aggregated_keywords_scores:
                final_keywords_with_scores = apply_final_filters(
                    aggregated_keywords_scores,
                    keyword_frequencies,
                    nlp_model,
                    args.pos_tags,
                    args.min_len,
                    args.min_doc_freq,
                    apply_pos_filter=not args.no_pos_filter
                )

                # Select final Top N from filtered list
                final_keywords_with_scores = final_keywords_with_scores[:args.final_top_n]

                if final_keywords_with_scores:
                    # Print Top N Final Keywords
                    print(f"\n--- Top {len(final_keywords_with_scores)} Filtered Domain Keywords (ranked by Max KeyBERT score) ---")
                    for term, score in final_keywords_with_scores:
                        print(f"{term} : {score:.4f}") # Score is Max KeyBERT relevance
                    print("-----------------------------------------------------------------------------")

                    # Cluster the final keywords
                    keyword_list_for_clustering = [term for term, _ in final_keywords_with_scores]
                    clustered_keywords = cluster_keywords(keyword_list_for_clustering, n_clusters=DEFAULT_N_CLUSTERS) # Use DEFINED N_CLUSTERS

                    if clustered_keywords:
                        print("\n--- Keyword Clusters (Top Keywords) ---")
                        print(json.dumps(clustered_keywords, indent=2))
                        print("---------------------------------------")
                    else:
                         logger.warning("Clustering did not produce results.")

                else:
                    print("No keywords remained after all filtering steps.")
            else:
                print("KeyBERT did not extract any keywords after aggregation.")
        else:
            print("No content retrieved from Weaviate.")

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print(f"❌ Failure: {str(e)}")
        sys.exit(1)
    finally:
        if client and client.is_connected():
            logger.info("Closing Weaviate client connection.")
            client.close()
