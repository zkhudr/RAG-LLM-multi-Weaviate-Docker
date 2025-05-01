Key System Interactions
Frontend (Alpine.js): Handles chat, file upload, config, and interacts with backend via REST endpoints.

Backend (Flask): Orchestrates pipeline, config, chat history, Weaviate state, and document processing.

Pipeline: Integrates LLM, retriever, embeddings, domain keywords, and centroid.

Retriever: Handles Weaviate client, vectorstore, embeddings, and context retrieval.

Ingestion: Handles document loading, extraction, storage, and incremental processing.

KeywordBuilder: Extracts and clusters keywords from Weaviate content.
