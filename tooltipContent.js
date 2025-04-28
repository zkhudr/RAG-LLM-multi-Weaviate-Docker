// static/tooltipContent.js
const tooltipContent = {
    // Model Section
    "LLM_TEMPERATURE": "Sampling temp (0-2). Low (<0.3) = deterministic/factual. High (>0.7) = creative/diverse. Rec: 0.3-0.7. Tune for task. Start low for stability.",
    "MAX_TOKENS": "Max output tokens from LLM. Hard limit on response length. Balance vs context window & latency. Rec: 512-1024. Ensure fits context.",
    "OLLAMA_MODEL": "Specify local Ollama model name (e.g., 'llama3', 'mistral'). Must be pulled via `ollama pull <model>`.",
    "EMBEDDING_MODEL": "Vector embedding model (e.g., 'nomic-embed-text', 'all-MiniLM-L6-v2'). Impacts retrieval quality & speed. Check dimensions compatibility.",
    "TOP_P": "Nucleus sampling (0-1). Alternative/combo with temp. Considers tokens within top P% probability mass. Rec: ~0.9. Less sensitive than temp.",
    "FREQUENCY_PENALTY": "Penalize token frequency (0-2). Reduces word/phrase repetition. Higher values = stronger penalty. Rec: 0.1-0.4.",
    "SYSTEM_MESSAGE": "LLM system prompt. Defines persona, task, constraints, output format. Crucial for instruction following & RAG integration.",
    "EXTERNAL_API_PROVIDER": "Select external LLM API provider ('openai', 'deepseek', 'anthropic', etc.). Requires corresponding API key in .env/secrets.",
    "EXTERNAL_API_MODEL_NAME": "Optional: Specify provider's model (e.g., 'gpt-4o', 'deepseek-chat'). Overrides provider default if set. Check provider docs for valid names.",

    // Retrieval Section
    "COLLECTION_NAME": "Target Weaviate collection/class name. Case-sensitive. Must match index schema.",
    "K_VALUE": "Top-K documents retrieved per query. More context vs prompt size/noise/latency. Rec: 5-10. Tune based on LLM ctx limit & answer quality.",
    "SCORE_THRESHOLD": "Min vector similarity score (0-1) for retrieved chunks. Filters irrelevant results post-retrieval. Rec: 0.6-0.75. Tune based on results.",
    "LAMBDA_MULTIPLIER": "MMR lambda (0-1). Balances relevance vs diversity. 1.0 = pure similarity, 0.0 = max diversity. Rec: 0.5-0.7 (bias relevance).",
    "SEARCH_TYPE": "Weaviate query method: 'similarity' (vector only), 'mmr' (similarity + diversity), 'hybrid' (vector + keyword/BM25 - *if implemented*).",
    "DOMAIN_SIMILARITY_THRESHOLD": "Vector similarity threshold (0-1) for initial query domain check (if enabled). Filters off-topic queries. Rec: 0.6-0.75.",
    "SPARSE_RELEVANCE_THRESHOLD": "Min scaled BM25 score (0-1) for sparse retrieval relevance (hybrid search). Corpus-dependent. Rec: ~0.1-0.2.",
    "FUSED_RELEVANCE_THRESHOLD": "Min combined score (0-1) after RRF/weighting dense & sparse results (hybrid search). Final relevance gate. Rec: ~0.3-0.4.",
    "SEMANTIC_WEIGHT": "Weight for dense (vector) search in hybrid fusion (e.g., RRF alpha). Rec: ~0.5-0.8. Often `semantic + sparse = 1.0`.",
    "SPARSE_WEIGHT": "Weight for sparse (keyword/BM25) search in hybrid fusion (e.g., RRF 1-alpha). Rec: ~0.2-0.5. Often `semantic + sparse = 1.0`.",
    "PERFORM_DOMAIN_CHECK": "Enable/disable initial query domain check using vector similarity. Keeps RAG focused on relevant topics. Recommended.",
    "WEAVIATE_HOST": "Weaviate instance hostname or IP address. Ensure accessible from app.",
    "WEAVIATE_HTTP_PORT": "Weaviate instance HTTP port (default 8080 or 8090). Match Weaviate config.",
    "WEAVIATE_GRPC_PORT": "Weaviate instance gRPC port (default 50051 or 50061). Match Weaviate config.",
    "retrieve_with_history": "Append chat history to user query before embedding for retrieval. Improves conversational context, can dilute query focus.",
    // --- Added/Updated Domain Related ---
    "DOMAIN_CENTROID": "Domain vector (`.npy`). Pre-calc semantic center. Used by `PERFORM_DOMAIN_CHECK` for query relevance via cosine similarity. Impacts retrieval filtering.",
    "DOMAIN_KEYWORDS": "Core keyword list (config/YAML). Defines static domain terms. Used in sparse part of `PERFORM_DOMAIN_CHECK`. Base for relevance tuning.",
    "AUTO_DOMAIN_KEYWORDS": "Keywords from builder script output. Dynamically generated. Supplements `DOMAIN_KEYWORDS` for sparse checks. Overwritten on rebuild.",
    "USER_ADDED_KEYWORDS": "Keywords manually added by user (via UI/config?). Supplements static/auto keywords for sparse checks. User-controlled additions.",
    // --- End Added/Updated ---

    // Security Section
    "SANITIZE_INPUT": "Enable basic input sanitization (potential security measure). Mitigates simple injection vectors. Recommended.",
    "RATE_LIMIT": "Max requests/user/minute. Protects backend resources/APIs. Rec: ~20-60 req/min. Adjust based on load & capacity.",
    "API_TIMEOUT": "Max seconds to wait for external LLM API response. Prevents indefinite hangs. Rec: 30-90s. Depends on model & task complexity.",
    "CACHE_ENABLED": "Enable simple in-memory cache. Returns cached response for *identical* input queries (incl. history). Reduces latency/cost for repeats.",

    // Document Section
    "CHUNK_SIZE": "Max tokens per text chunk for indexing. Balance context vs retrieval precision & embedding model limits. Rec: 512-1024. Key param.",
    "CHUNK_OVERLAP": "Token overlap between adjacent chunks. Maintains context across boundaries. Rec: 10-20% of Chunk Size (e.g., 50-150).",
    "FILE_TYPES": "Allowed file extensions for ingestion (comma-sep, e.g., '.pdf,.txt,.md'). Filter for parser.",
    "PARSE_TABLES": "Attempt structured table extraction during document parsing (e.g., via Unstructured). Useful for tabular data, adds processing time.",
    "GENERATE_SUMMARY": "Generate LLM summary per document during ingestion (store as metadata?). Adds significant processing time/cost.",
    // --- Added ---
    "DOCUMENT_DIRECTORY": "Ingestion source path. App scans here for files (PDF, TXT...). Affects *what* gets ingested. Check server permissions.",
    // --- End Added ---

    // Pipeline Section
    "max_history_turns": "Max recent Q&A pairs from history included in LLM prompt. Balances conversational context vs token limit. Rec: 3-5 pairs.",

    // Domain Keywords Extraction (Assuming these settings relate to the builder script UI/form)
    "SENTENCE_TRANSFORMER_MODEL": "Alias for EMBEDDING_MODEL. Specifies model for keyword extraction vectorization (if used). Needs to match retrieval embedder for centroid use.",
    "KEYWORDS_PER_DOCUMENT": "Max candidate keywords extracted *per document* before global filtering/ranking. Rec: 10-20. Based on content density.",
    "FINAL_KEYWORDS_COUNT": "Target number of *unique* keywords across corpus after ranking/MMR. Limits final keyword set size. Rec: 500-5000+.",
    "MINIMUM_DOCUMENT_FREQUENCY": "Min document frequency for keyword inclusion. Absolute count (e.g., 3) or float ratio (e.g., 0.01 = 1%). Filters rare/noisy terms. Rec: 2-5 or 0.005-0.01.",
    "DIVERSITY": "MMR lambda for final keyword selection (0-1). Balances relevance vs diversity *among final keywords*. 0=most relevant, 1=max diverse. Rec: 0.5-0.8.",
    "DISABLE_POS_FILTERING": "Disable filtering keywords by Part-of-Speech tags (e.g., nouns/adjectives). Enable if crucial domain terms (acronyms, codes) are missed."
};
