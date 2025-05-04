// static/tooltipContent.js
const tooltipContent = {
    // Model Section
    "LLM_TEMPERATURE": "Sampling temperature T ∈ [0,2]. Controls softmax sharpness: P(wᵢ) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T). Lower T → argmax-like; higher T → uniform random :contentReference[oaicite:0]{index=0}",
    "MAX_TOKENS": "Hard cap on output length in tokens. Enforced as Σₜ 1 ≤ max_tokens. Balances context window & latency—overflow truncates at model level.",
    "OLLAMA_MODEL": "Local Ollama model ID (e.g. 'llama3', 'mistral'). Must be fetched via `ollama pull <model>` before instantiation.",
    "EMBEDDING_MODEL": "Text→vector encoder (e.g. 'nomic-embed-text', 'all-MiniLM-L6-v2'). Embedding dim (d) impacts memory & cosine sims :contentReference[oaicite:1]{index=1}",
    "TOP_P": "Nucleus sampling (p ∈ [0,1]): pick minimal V s.t. Σ_{w∈V}P(w) ≥ p, then renormalize P′(w)=P(w)/Σ_{v∈V}P(v) :contentReference[oaicite:2]{index=2}",
    "FREQUENCY_PENALTY": "Logit adjustment: logit′(w)=logit(w)−α·count(w), with α ∈ [0,2]. Higher α penalizes repeated tokens :contentReference[oaicite:3]{index=3}",
    "SYSTEM_MESSAGE": "Root prompt (persona/task/etc.) injected at system level. Defines service-level constraints & RAG integration.",
    "EXTERNAL_API_PROVIDER": "Select LLM API: 'openai', 'anthropic', etc. Ensure corresponding API key in env.",
    "EXTERNAL_API_MODEL_NAME": "Override default model (e.g. 'gpt-4o', 'deepseek-chat'). Must match provider’s endpoint spec.",

    // Retrieval Section
    "COLLECTION_NAME": "Target Weaviate class/collection. Case-sensitive; must match schema definition.",
    "K_VALUE": "Retrieve top-K docs by descending similarity. K ∈ ℕ; trade-off: more context vs. latency & noise.",
    "SCORE_THRESHOLD": "Filter out chunks with similarity < τ (τ ∈ [0,1]). Typical τ = 0.6–0.75 to prune off-topic hits.",
    "LAMBDA_MULTIPLIER": "MMR λ ∈ [0,1]. Rank by: argmax_{d ∈ D\R}[λ·Sim(q,d) − (1−λ)·max_{r ∈ R}Sim(d,r)] :contentReference[oaicite:4]{index=4}",
    "SEARCH_TYPE": "Retrieval mode: 'similarity' (vector-only), 'mmr' (adds diversity term), or 'hybrid' (vector + BM25).",
    "DOMAIN_SIMILARITY_THRESHOLD": "Initial filter: CosineSim(q, centroid) = (q·c)/(|q||c|) ≥ θ. Prevent OOD queries :contentReference[oaicite:5]{index=5}",
    "SPARSE_RELEVANCE_THRESHOLD": "In hybrid, drop docs with BM25_norm < σ. Typical σ ≈ 0.1–0.2.",
    "FUSED_RELEVANCE_THRESHOLD": "After combining dense + sparse: final score ≥ φ (φ ∈ [0,1]).",
    "SEMANTIC_WEIGHT": "Weight α for dense in fusion: score = α·dense + (1−α)·sparse.",
    "SPARSE_WEIGHT": "Weight (1−α) for sparse in fusion; semantic_weight + sparse_weight = 1.",
    "PERFORM_DOMAIN_CHECK": "Boolean toggle for pre-retrieval domain centroid check.",
    "PERFORM_TECHNICAL_VALIDATION": "Bypass keyword-domain filters; accept raw pipeline output.",
    "WEAVIATE_HOST": "Hostname/IP of Weaviate instance. Ensure network reachability.",
    "WEAVIATE_HTTP_PORT": "HTTP port (default 8080/8090). Matches your Weaviate config.",
    "WEAVIATE_GRPC_PORT": "gRPC port (default 50051/50061). Matches your Weaviate config.",
    "retrieve_with_history": "If true, append last N chat turns to query before embedding—boosts context but may blur intent.",

    // Domain Vector Settings
    "DOMAIN_CENTROID": "Filepath (.npy) for precomputed domain centroid vector. Used for cosine relevance gating.",
    "DOMAIN_KEYWORDS": "Manually curated keyword list for sparse domain checks.",
    "AUTO_DOMAIN_KEYWORDS": "Builder-generated keywords; overwritten on rebuild.",
    "USER_ADDED_KEYWORDS": "Extra keywords from UI/config to augment domain lexicon.",

    // Security Section
    "SANITIZE_INPUT": "Enable basic input sanitization to mitigate injection attacks. Recommended=true.",
    "RATE_LIMIT": "Max requests per user per minute (e.g. 20–60). Protects API/backend stability.",
    "API_TIMEOUT": "Timeout (s) for external LLM API calls. Common: 30–90s.",
    "CACHE_ENABLED": "Enable in-memory caching of identical queries (including history) for speed.",

    // Document Section
    "CHUNK_SIZE": "Max tokens per text chunk during indexing. Balance retrieval granularity vs. context. Common: 512–1024.",
    "CHUNK_OVERLAP": "Token overlap between chunks. Ensures context continuity. Typical: 10–20% of chunk size.",
    "FILE_TYPES": "Comma-separated list of extensions to ingest (e.g. '.pdf,.txt,.md').",
    "PARSE_TABLES": "Attempt structured table extraction. Useful for data-rich docs; incurs extra parse time.",
    "GENERATE_SUMMARY": "Auto-summarize each doc on ingest. Speeds up previews but adds LLM calls.",
    "DOCUMENT_DIRECTORY": "Filesystem path for ingestion. App must have read perms.",

    // Pipeline Section
    "max_history_turns": "Include last N Q&A pairs in each prompt. N ∈ ℕ; typical: 3–5 to fit context window.",

    // Keyword Extraction Section
    "SENTENCE_TRANSFORMER_MODEL": "Embedding model alias for keyword extraction; should match EMBEDDING_MODEL.",
    "KEYWORDS_PER_DOCUMENT": "Max candidates per doc before global MMR filtering. Typical: 10–20.",
    "FINAL_KEYWORDS_COUNT": "Target unique keywords across corpus. Adjust to 500–5000+ based on size.",
    "MINIMUM_DOCUMENT_FREQUENCY": "Drop keywords with doc_freq < df_min (absolute count or ratio, e.g. 3 or 0.01).",
    "DIVERSITY": "In keyword MMR, λ ∈ [0,1]: balances relevance vs. diversity. Typical: 0.5–0.8.",
    "DISABLE_POS_FILTERING": "Disable POS-based pruning to keep acronyms/codes that might be filtered out."
};
