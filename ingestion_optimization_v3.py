"""
ingestion_optimization.py

This file serves as a CLI and testing tool for your ingestion, cleaning, ranking,
keyword extraction, and generation pipelines. It includes a simple CSV logging mechanism
to record each experiment run (options 1-6) with their parameters and outputs.
"""

import os
import sys
import csv
import logging
import ingest_docs_v3
from datetime import datetime

# Import processing functions from your modules
from ingest_docs_v3 import process_documents as ingest_process_documents
from TF_IDF_BERT_Clean import process_documents as tfidf_bert_process_documents
from text_ranker import process_documents as text_ranker_process_documents
from build_domain_keywords import build_domain_keywords
from pipeline import UnifiedPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def log_experiment(experiment_name, parameters, outputs, log_file="experiment_log.csv"):
    """
    Logs experiment details to a CSV file.
    
    :param experiment_name: A short name for the experiment run.
    :param parameters: A dictionary of parameters used in the run.
    :param outputs: A dictionary of key outputs (e.g. top keywords, feature shapes).
    :param log_file: CSV file path to store logs.
    """
    file_exists = os.path.exists(log_file)
    fieldnames = ["timestamp", "experiment_name", "parameters", "outputs"]
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "parameters": parameters,
            "outputs": outputs
        })


def prompt_parameters():
    """
    Prompts the user for pipeline parameters.

    General parameters:
      - Data directory (default: './data')

    Mode-specific parameters:
      For modes 3 & 4 (Text Ranker): Chunk size and overlap.
      For modes 2 & 4 (TF-IDF-BERT Clean): TF-IDF parameters (max_features, min_df, ngram range, PCA settings).
      For mode 5 (Domain Keyword Extraction): Top N keywords.
      For mode 6 (Generation Test): Generation parameters (Temperature, Max Tokens, Top_p, Frequency Penalty).
    """
    print("=== Ingestion, Cleaning & Generation Pipeline Optimization ===")
    data_dir = input("Enter data directory (default './data'): ").strip() or "./data"
    
    print("\nSelect processing mode:")
    print("  1. Ingestion Only (full ingestion with vector store update)")
    print("  2. TF-IDF-BERT Clean (text cleaning & feature extraction)")
    print("  3. Text Ranker (chunking & ranking)")
    print("  4. Combined Processing (TF-IDF-BERT Clean + Text Ranker)")
    print("  5. Domain Keyword Extraction (top domain keywords)")
    print("  6. Generation Test (end-to-end retrieval + LLM response)")
    mode = input("Enter choice (1-6, default 4): ").strip() or "4"
    
    # Default extra parameters
    chunk_size = 512
    chunk_overlap = 128
    max_features = 100
    min_df = 2
    ngram_low, ngram_high = 1, 2
    apply_pca = True
    pca_components = 100
    top_n = 100
    gen_temp = 0.7
    gen_max_tokens = 1024
    gen_top_p = 0.9
    gen_freq_penalty = 0.1

    # For Text Ranker (mode 3 & 4) prompt for chunking parameters.
    if mode in ["3", "4"]:
        try:
            chunk_size = int(input("Enter chunk size for Text Ranker (default 512): ").strip() or "512")
        except ValueError:
            print("Invalid chunk size. Using default 512.")
            chunk_size = 512
        try:
            chunk_overlap = int(input("Enter chunk overlap for Text Ranker (default 128): ").strip() or "128")
        except ValueError:
            print("Invalid chunk overlap. Using default 128.")
            chunk_overlap = 128

    # For TF-IDF-BERT Clean (mode 2 & 4) prompt for TF-IDF parameters.
    if mode in ["2", "4"]:
        try:
            max_features = int(input("Enter max_features for TF-IDF (default 100): ").strip() or "100")
        except ValueError:
            print("Invalid input. Using default 100.")
            max_features = 100
        try:
            min_df = int(input("Enter min_df for TF-IDF (default 2): ").strip() or "2")
        except ValueError:
            print("Invalid input. Using default 2.")
            min_df = 2
        ngram_range_input = input("Enter ngram range as two numbers separated by a comma (default 1,2): ").strip() or "1,2"
        try:
            ngram_low, ngram_high = map(int, ngram_range_input.split(","))
        except Exception:
            print("Invalid input. Using default (1,2).")
            ngram_low, ngram_high = 1, 2
        apply_pca_input = input("Apply PCA? (Y/N, default Y): ").strip().upper() or "Y"
        apply_pca = (apply_pca_input == "Y")
        if apply_pca:
            try:
                pca_components = int(input("Enter number of PCA components (default 100): ").strip() or "100")
            except ValueError:
                print("Invalid input. Using default 100.")
                pca_components = 100

    # For Domain Keyword Extraction (mode 5)
    if mode == "5":
        try:
            top_n = int(input("Enter number of top domain keywords to extract (default 100): ").strip() or "100")
        except ValueError:
            print("Invalid input. Using default 100.")
            top_n = 100

    # For Generation Test (mode 6)
    if mode == "6":
        query = input("Enter query for Generation Test (default 'What is SCADA?'): ").strip() or "What is SCADA?"
        try:
            gen_temp = float(input("Enter Temperature (default 0.7): ").strip() or "0.7")
            gen_max_tokens = int(input("Enter Max Tokens (default 1024): ").strip() or "1024")
            gen_top_p = float(input("Enter Top_p (default 0.9): ").strip() or "0.9")
            gen_freq_penalty = float(input("Enter Frequency Penalty (default 0.1): ").strip() or "0.1")
        except ValueError:
            print("Invalid generation parameters. Using defaults.")
            gen_temp, gen_max_tokens, gen_top_p, gen_freq_penalty = 0.7, 1024, 0.9, 0.1

    params = {
        "data_directory": data_dir,
        "mode": mode,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_features": max_features,
        "min_df": min_df,
        "ngram_range": (ngram_low, ngram_high),
        "apply_pca": apply_pca,
        "pca_components": pca_components,
        "top_n": top_n,
        # Generation parameters (used only in mode 6)
        "gen_query": query if mode == "6" else None,
        "gen_temperature": gen_temp,
        "gen_max_tokens": gen_max_tokens,
        "gen_top_p": gen_top_p,
        "gen_frequency_penalty": gen_freq_penalty
    }
    return params

def option_ingestion(data_directory):
    print(f"\n[Mode 1] Running full ingestion pipeline from {data_directory}...")
    try:
        ingest_process_documents(data_directory)
        outputs = {"note": "Ingestion complete; vector store updated."}
        log_experiment("Ingestion_Full", {"data_directory": data_directory}, outputs)
        print("Ingestion complete.")
    except Exception as e:
        print(f"Ingestion failed: {e}")

def option_tfidf_bert(data_directory, params):
    print(f"\n[Mode 2] Running TF-IDF-BERT Clean on documents in {data_directory}...")
    try:
        # Pass the TF-IDF parameters to the process_documents function.
        keywords, features = tfidf_bert_process_documents(
            data_directory,
            max_features=params["max_features"],
            min_df=params["min_df"],
            ngram_range=params["ngram_range"],
            apply_pca=params["apply_pca"],
            pca_components=params["pca_components"]
        )
        print("TF-IDF-BERT Clean Results:")
        print("Top Keywords:", keywords)
        print("Feature Matrix Shape:", features.shape)
        outputs = {"top_keywords": list(keywords), "features_shape": features.shape}
        log_experiment("TFIDF_BERT_Clean", {"data_directory": data_directory, **params}, outputs)
    except Exception as e:
        print(f"TF-IDF-BERT Clean failed: {e}")

def option_text_ranker(data_directory, chunk_size, chunk_overlap):
    print(f"\n[Mode 3] Running Text Ranker on documents in {data_directory} with chunk size {chunk_size} and overlap {chunk_overlap}...")
    try:
        keywords, features = text_ranker_process_documents(data_directory, chunk_size, chunk_overlap)
        print("Text Ranker Results:")
        print("Top Keywords:", keywords)
        print("Feature Matrix Shape:", features.shape)
        outputs = {"top_keywords": list(keywords), "features_shape": features.shape}
        log_experiment("Text_Ranker", {"data_directory": data_directory, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}, outputs)
    except Exception as e:
        print(f"Text Ranker failed: {e}")

def option_combined(data_directory, chunk_size, chunk_overlap, params):
    print(f"\n[Mode 4] Running Combined Processing on documents in {data_directory}...")
    try:
        print("\n--- TF-IDF-BERT Clean ---")
        keywords1, features1 = tfidf_bert_process_documents(
            data_directory,
            max_features=params["max_features"],
            min_df=params["min_df"],
            ngram_range=params["ngram_range"],
            apply_pca=params["apply_pca"],
            pca_components=params["pca_components"]
        )
        print("Top Keywords:", keywords1)
        print("Feature Matrix Shape:", features1.shape)
    except Exception as e:
        print(f"TF-IDF-BERT Clean failed: {e}")
    try:
        print("\n--- Text Ranker ---")
        keywords2, features2 = text_ranker_process_documents(data_directory, chunk_size, chunk_overlap)
        print("Top Keywords:", keywords2)
        print("Feature Matrix Shape:", features2.shape)
        outputs = {
            "TFIDF_BERT": {"top_keywords": list(keywords1), "features_shape": features1.shape},
            "Text_Ranker": {"top_keywords": list(keywords2), "features_shape": features2.shape},
        }
        log_experiment("Combined_Processing", {"data_directory": data_directory, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **params}, outputs)
    except Exception as e:
        print(f"Text Ranker failed: {e}")

def option_domain_keywords(data_directory, top_n):
    print(f"\n[Mode 5] Extracting top {top_n} domain keywords from documents in {data_directory}...")
    try:
        keywords = build_domain_keywords(data_directory, top_n)
        print("Top Domain Keywords:")
        for term, score in keywords:
            print(f"{term}: {score:.4f}")
        outputs = {"domain_keywords": [(term, score) for term, score in keywords]}
        log_experiment("Domain_Keyword_Extraction", {"data_directory": data_directory, "top_n": top_n}, outputs)
    except Exception as e:
        print(f"Domain Keyword Extraction failed: {e}")

def option_generation_test(gen_query, gen_temp, gen_max_tokens, gen_top_p, gen_freq_penalty):
    print(f"\n[Mode 6] Running Generation Test for query: '{gen_query}'")
    try:
        pipeline = UnifiedPipeline()
        # Update generation parameters on the pipeline config:
        pipeline.cfg.model.LLM_TEMPERATURE = gen_temp
        pipeline.cfg.model.MAX_TOKENS = gen_max_tokens
        pipeline.cfg.model.TOP_P = gen_top_p
        pipeline.cfg.model.FREQUENCY_PENALTY = gen_freq_penalty
        
        result = pipeline.generate_response(gen_query)
        if "error" in result:
            print("Error:", result.get("error"))
            if "details" in result:
                print("Details:", result.get("details"))
        else:
            print("\n--- Generation Test Result ---")
            print("Response Source:", result.get("source", "N/A").upper())
            print("Model:", result.get("model", "N/A"))
            print("Response:", result.get("response", "No response"))
        outputs = {"generation_response": result}
        log_experiment("Generation_Test", {
            "query": gen_query,
            "temperature": gen_temp,
            "max_tokens": gen_max_tokens,
            "top_p": gen_top_p,
            "frequency_penalty": gen_freq_penalty
        }, outputs)
    except Exception as e:
        print(f"Generation Test failed: {e}")

def main():
    params = prompt_parameters()
    data_dir = params["data_directory"]
    mode = params["mode"]

    if mode == "1":
        option_ingestion(data_dir)
    elif mode == "2":
        option_tfidf_bert(data_dir, params)
    elif mode == "3":
        option_text_ranker(data_dir, params["chunk_size"], params["chunk_overlap"])
    elif mode == "4":
        option_combined(data_dir, params["chunk_size"], params["chunk_overlap"], params)
    elif mode == "5":
        option_domain_keywords(data_dir, params["top_n"])
    elif mode == "6":
        option_generation_test(params["gen_query"], params["gen_temperature"], params["gen_max_tokens"], params["gen_top_p"], params["gen_frequency_penalty"])
    else:
        print("Invalid choice. No processing executed.")
    print("\nExperiment logged. Check experiment_log.csv for details.")

if __name__ == "__main__":
    main()
