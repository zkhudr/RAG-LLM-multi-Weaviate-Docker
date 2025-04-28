# pipeline.py (Corrected for Chat History)

import re
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple, List # Ensure List, Dict, Optional are imported
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming these are correctly importable from your project structure
try:
    from langchain_ollama import OllamaLLM
    from config import cfg # Use 'cfg' directly if it's the loaded instance
    from retriever import TechnicalRetriever
    imports_ok = True
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import core pipeline dependencies: {e}", exc_info=True)
    imports_ok = False
    # Define dummy classes/objects to prevent immediate crash if possible, but log error
    class OllamaLLM: pass
    class TechnicalRetriever: pass
    class cfg: # Dummy config
        class model: OLLAMA_MODEL="dummy"; LLM_TEMPERATURE=0.7; MAX_TOKENS=512; SYSTEM_MESSAGE=""
        class security: SANITIZE_INPUT=True; DEEPSEEK_API_KEY=""; API_TIMEOUT=20; CACHE_ENABLED=False
        class retrieval: PERFORM_DOMAIN_CHECK=False; DOMAIN_SIMILARITY_THRESHOLD=0.6; SPARSE_RELEVANCE_THRESHOLD=0.1; FUSED_RELEVANCE_THRESHOLD=0.4; SEMANTIC_WEIGHT=0.7; SPARSE_WEIGHT=0.3
        class paths: DOMAIN_CENTROID_PATH="./dummy_centroid.npy"
        class env: merged_keywords=[]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndustrialAutomationPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not imports_ok:
             self.logger.critical("Pipeline cannot initialize due to missing imports.")
             # Prevent further initialization if core components missing
             raise ImportError("Core pipeline dependencies failed to import.")

        self.cfg = cfg
        try:
            self.local_llm = OllamaLLM(
                model=self.cfg.model.OLLAMA_MODEL,
                temperature=self.cfg.model.LLM_TEMPERATURE,
                # Removed num_predict, let Ollama handle context/token limits unless specifically needed & supported
            )
            self.retriever = TechnicalRetriever()
            self.embeddings = self.retriever.embeddings
        except Exception as init_e:
            self.logger.critical(f"Failed to initialize LLM or Retriever: {init_e}", exc_info=True)
            raise # Stop if essential components fail

        # Use a set for fast sparse check - Ensure merged_keywords property exists in config
        self.domain_keywords_set = set(getattr(self.cfg.env, 'merged_keywords', []))

        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- Load Domain Centroid ---
        self.domain_centroid = self._load_domain_centroid()
        if self.domain_centroid is None and self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
            logger.warning("Domain centroid not loaded. Semantic relevance check will be skipped.")
            # Optionally disable the check entirely if centroid is crucial
            # self.cfg.retrieval.PERFORM_DOMAIN_CHECK = False
        # -----------------------------

        if not self.domain_keywords_set and self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
            logger.warning("Domain keywords list is EMPTY during pipeline initialization!")
        else:
            logger.info(f"Pipeline initialized with {len(self.domain_keywords_set)} domain keywords.")

    def _load_domain_centroid(self) -> Optional[np.ndarray]:
        """Loads the pre-calculated domain centroid vector."""
        try:
            centroid_path = self.cfg.paths.DOMAIN_CENTROID_PATH
            if os.path.exists(centroid_path):
                centroid = np.load(centroid_path)
                logger.info(f"Loaded domain centroid vector from {centroid_path} with shape {centroid.shape}")
                # Reshape to (1, embedding_dim) for cosine_similarity
                return centroid.reshape(1, -1)
            else:
                logger.error(f"Domain centroid file not found at: {centroid_path}. Run calculate_centroid.py.")
                return None
        except Exception as e:
            logger.error(f"Failed to load domain centroid: {e}", exc_info=True)
            return None

    def _sanitize_input(self, query: str) -> str:
        """Basic sanitization of input query."""
        self.logger.debug(f"Sanitizing query (original): '{query[:100]}...'")
        # Remove HTML-like tags
        query = re.sub(r'<[^>]*>', '', query)
        # Remove control characters (excluding newline, tab, carriage return)
        query = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', query)
        # Keep only reasonably safe characters: letters, numbers, whitespace, basic punctuation
        query = re.sub(r'[^\w\s\-\.,;!?\'"]', '', query, flags=re.UNICODE)
        sanitized = query.strip()
        self.logger.debug(f"Sanitizing query (result): '{sanitized[:100]}...'")
        return sanitized

    # === Cache functions (Consider disabling/modifying for chat history) ===
    def _get_cache_key(self, content: str) -> str:
        """Generates a cache key from content string."""
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Loads response from cache if available using pre-generated key."""
        if not self.cfg.security.CACHE_ENABLED: return None # Skip if disabled

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.logger.debug(f"Cache hit for key: {cache_key} Path: {cache_path}")
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Cache file {cache_path} is corrupted. Removing.")
                try: os.remove(cache_path)
                except OSError as e: self.logger.error(f"Failed to remove corrupted cache file {cache_path}: {e}")
            except Exception as e:
                self.logger.error(f"Failed to read cache file {cache_path}: {e}")
        else:
            self.logger.debug(f"Cache miss for key: {cache_key}...")
        return None

    def _save_to_cache(self, cache_key: str, response: Dict):
        """Saves response to cache using pre-generated key."""
        if not self.cfg.security.CACHE_ENABLED: return # Skip if disabled

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
            self.logger.debug(f"Saved response to cache: {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to write to cache file {cache_path}: {e}")
    # === End Cache Functions ===


    def _calculate_semantic_score(self, query: str) -> float:
        """Calculates cosine similarity between query embedding and domain centroid."""
        if self.domain_centroid is None:
            logger.warning("Cannot calculate semantic score, domain centroid not loaded.")
            return 0.0
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_vector, self.domain_centroid)[0][0]
            similarity = max(0.0, min(1.0, similarity)) # Clip
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic score for query '{query[:50]}...': {e}", exc_info=True)
            return 0.0

    def _calculate_sparse_score(self, query_terms: List[str]) -> float:
        """Calculates keyword overlap ratio between query terms and domain keywords."""
        if not self.domain_keywords_set or not query_terms:
            self.logger.debug("Sparse Check: No keywords or terms, returning 0.0")
            return 0.0

        common_words = {"the", "a", "an", "is", "in", "of", "to", "for", "with", "what", "how", "why", "explain", "describe", "compare", "difference"}
        cleaned_query_terms = [
            re.sub(r'[.,!?;:]+$', '', term) # Remove trailing punctuation
            for term in query_terms
            if term not in common_words and len(term) > 1
        ]

        if not cleaned_query_terms:
            self.logger.debug("Sparse Check: No valid terms after cleaning/filtering, returning 0.0")
            return 0.0

        match_count = sum(1 for term in cleaned_query_terms if term in self.domain_keywords_set)
        overlap_ratio = match_count / len(cleaned_query_terms)
        self.logger.debug(f"Sparse Check: Cleaned Terms={cleaned_query_terms}, Matches={match_count}, Ratio={overlap_ratio:.3f}")
        return overlap_ratio

    def _check_domain_relevance_hybrid(self, query: str, query_terms: List[str]) -> bool:
        """Performs hybrid relevance check using semantic and sparse scores."""
        semantic_score = self._calculate_semantic_score(query)
        sparse_score = self._calculate_sparse_score(query_terms)

        fused_score = (semantic_score * self.cfg.retrieval.SEMANTIC_WEIGHT) + \
                      (sparse_score * self.cfg.retrieval.SPARSE_WEIGHT)

        semantic_relevant = semantic_score >= self.cfg.retrieval.DOMAIN_SIMILARITY_THRESHOLD
        sparse_relevant = sparse_score >= self.cfg.retrieval.SPARSE_RELEVANCE_THRESHOLD
        fused_relevant = fused_score >= self.cfg.retrieval.FUSED_RELEVANCE_THRESHOLD

        # Simple fused threshold check:
        is_relevant = fused_relevant

        self.logger.info(
            f"Hybrid Relevance Check: Semantic={semantic_score:.3f} (Thresh={self.cfg.retrieval.DOMAIN_SIMILARITY_THRESHOLD}, Pass={semantic_relevant}), "
            f"Sparse={sparse_score:.3f} (Thresh={self.cfg.retrieval.SPARSE_RELEVANCE_THRESHOLD}, Pass={sparse_relevant}), "
            f"Fused={fused_score:.3f} (Thresh={self.cfg.retrieval.FUSED_RELEVANCE_THRESHOLD}, Pass={is_relevant})"
        )
        return is_relevant

    # === NEW: Helper to format history for LLM prompt ===
    def _format_chat_history_for_prompt(self, chat_history: Optional[List[Dict]]) -> str:
        """ Formats chat history into a simple string for the LLM prompt, respecting max_history_turns. """
        if not chat_history:
            return ""
        
        formatted_history = ""  # Initialize here
        
        # Get max history turns from config
        max_turns = self.cfg.pipeline.max_history_turns
        
        # Calculate how many messages to include (2 messages per turn - user & assistant)
        max_messages = max_turns * 2
        
        # Get the appropriate slice of history
        if len(chat_history) > max_messages + 1:  # +1 to account for latest message
            history_to_use = chat_history[-max_messages-1:-1]  # Include up to max_messages, excluding latest
        else:
            history_to_use = chat_history[:-1]  # Use all previous if fewer than max
        
        # Debug logging
        self.logger.debug(f"Formatting {len(history_to_use)} messages from history (max_turns={max_turns})")
        
        for message in history_to_use:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "").strip()
            if content:  # Skip empty messages
                formatted_history += f"{role}:\n{content}\n\n"
        
        return formatted_history.strip()

    # === NEW: Helper to format history for LLM API ===
    def _format_chat_history_for_api(self, chat_history: Optional[List[Dict]]) -> List[Dict]:
        """ Formats chat history into the list format expected by API, respecting max_history_turns. """
        if not chat_history:
            return []
        
        # Get max history turns from config
        max_turns = self.cfg.pipeline.max_history_turns
        
        # Calculate how many messages to include (2 messages per turn - user & assistant)
        max_messages = max_turns * 2
        
        # Get the appropriate slice of history
        if len(chat_history) > max_messages:
            history_to_use = chat_history[-max_messages-1:-1]  # Include up to max_messages, excluding latest
        else:
            history_to_use = chat_history[:-1]  # Use all previous messages if fewer than max
        
        api_history = []
        for msg in history_to_use:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and content:  # Filter for valid roles/content
                api_history.append({"role": role, "content": content})
        
        return api_history

        # === MODIFIED: generate_response ===
    def generate_response(self, query: str, chat_history: Optional[List[Dict]] = None) -> Dict:
        """
        Generates response using context and optional chat history, with history-aware
        caching, retrieval, and optional response validation.
        """
        self.logger.info(f"Generating response with max_history_turns={self.cfg.pipeline.max_history_turns}, " 
                 f"history length: {len(chat_history) if chat_history else 0}")
        # --- History-Aware Caching Check ---
        # Get max history turns from config
        max_turns = self.cfg.pipeline.max_history_turns

        # Calculate how many messages to include
        max_messages = max_turns * 2

        # Get the appropriate slice of history for caching
        history_for_cache = chat_history[-max_messages-1:-1] if chat_history and len(chat_history) > max_messages else chat_history[:-1] if chat_history else None

        history_for_cache_key = self._format_chat_history_for_api(history_for_cache)
        history_str_for_key = json.dumps(history_for_cache_key) if history_for_cache_key else ""
        cache_key_content = f"{query}_{history_str_for_key}"
        cache_key = self._get_cache_key(cache_key_content)

        cached = self._load_from_cache(cache_key)
        if cached:
            self.logger.info("Returning CACHED response (history-aware key used)")
            cached.setdefault('response', '')
            cached.setdefault('source', 'cache')
            cached.setdefault('model', 'cache')
            cached.setdefault('context', '')
            cached.setdefault('error', False)
            cached['timestamp'] = datetime.now().isoformat()
            return cached
        # --- End Caching Check ---

        try:
            # --- Sanitization ---
            if self.cfg.security.SANITIZE_INPUT:
                sanitized_query = self._sanitize_input(query)
                if not sanitized_query or len(sanitized_query) < 3:
                    self.logger.warning("Query failed sanitization or was too short.")
                    return self._format_response("Query too short or invalid after sanitization", "validation", context=None, error=True)
            else:
                sanitized_query = query

            query_terms = [t for t in sanitized_query.lower().split() if len(t) > 1]

            # --- Domain Check ---
            if self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
                domain_match = self._check_domain_relevance_hybrid(sanitized_query, query_terms)
                if not domain_match:
                    self.logger.info("Query deemed outside domain scope by hybrid check.")
                    return self._format_response("Query appears to be outside the scope of industrial automation.", "domain_check", context=None)
                self.logger.info("Query passed hybrid domain relevance check.")
            else:
                self.logger.info("Domain relevance check skipped by configuration.")

            # --- History-Aware Context Retrieval ---
            # --- History-Aware Context Retrieval ---
            retrieval_query = sanitized_query
            if self.cfg.retrieval.retrieve_with_history and chat_history:
                # Get max history turns from config
                max_turns = self.cfg.pipeline.max_history_turns
                
                # Calculate how many messages to include
                max_messages = max_turns * 2
                
                # Get the appropriate slice of history for retrieval context
                history_for_retrieval = chat_history[-max_messages-1:-1] if len(chat_history) > max_messages else chat_history[:-1]
                
                last_user_message_content = ""
                for i in range(len(history_for_retrieval) - 1, -1, -1):
                    if history_for_retrieval[i].get("role") == "user":
                        last_user_message_content = history_for_retrieval[i].get("content", "")
                        break
                
                if last_user_message_content:
                    retrieval_query = f"Previous user question context: {last_user_message_content}\n\nCurrent user question: {sanitized_query}"
                    self.logger.info(f"Retrieving context using history-aware query: '{retrieval_query[:150]}...'")
                else:
                    self.logger.info("retrieve_with_history enabled, but no previous user message found.")
            else:
                self.logger.info(f"Retrieving context using latest query only: '{retrieval_query[:100]}...'")

            context = self.retriever.get_context(retrieval_query)
            if not context:
                self.logger.warning(f"No context found for retrieval query: {retrieval_query[:50]}...")

            # --- Response Generation (Pass History) ---
            local_response = self._generate_local_response(sanitized_query, context, chat_history)

            api_response = None
            provider = self.cfg.model.EXTERNAL_API_PROVIDER.lower()
            api_key_exists = False
            if provider == 'deepseek' and self.cfg.security.DEEPSEEK_API_KEY:
                api_key_exists = True
                api_response = self.call_deepseek_api(sanitized_query, context, chat_history)
            elif provider == 'openai' and self.cfg.security.OPENAI_API_KEY:
                self.logger.warning("OpenAI provider selected but call_openai_api not implemented yet.")
            elif provider != 'none':
                self.logger.warning(f"External provider '{provider}' configured but no key/call function.")

            merged_response = self.merge_responses(local_response, api_response)

            # --- Optional Validation --- START OF COMPLETED SECTION ---
            is_valid = self.validate_technical_response(merged_response)
            self.logger.info(f"Response validation result: {is_valid}")

            validation_failed_source = None # Flag to potentially override source later
            if not is_valid:
                self.logger.warning(f"Generated response failed technical validation. Original response snippet: '{merged_response[:100]}...'")
                # Option A: Replace the response
                merged_response = "The generated response may not be directly related to industrial automation or lacks specific technical terms. Please clarify or ask about a specific technical topic."
                validation_failed_source = "validation_failed" # Set a flag/source override
            # --- Optional Validation --- END OF COMPLETED SECTION ---

            # --- Format final response ---
            source = "local_llm"
            if api_response and provider != 'none':
                source = provider
            elif api_key_exists and not api_response and provider != 'none':
                source = f"{provider}_api_failed_fallback_local"

            # Override source if validation failed and we replaced the message
            if validation_failed_source:
                source = validation_failed_source

            final_result = self._format_response(
                response=merged_response,             # Use potentially modified response
                source=source,                         # Use potentially modified source
                context=context if is_valid else "",   # Don't show context if validation failed
                error=False                            # Not treating validation failure as a hard error
            )

            # --- Caching Save ---
            # Only cache responses that passed validation
            if is_valid:
                self._save_to_cache(cache_key, final_result) # Use history-aware key
            else:
                self.logger.info("Skipping cache save for validation-failed response.")

            return final_result

        except Exception as e:
            self.logger.exception(f"Pipeline error during generate_response for query '{query[:50]}...': {str(e)}")
            return self._format_response(
                response=f"Sorry, an internal error occurred processing your request.",
                source="pipeline_error",
                context=None,
                error=True
            )

    # === MODIFIED: _generate_local_response ===
    def _generate_local_response(self, query: str, context: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Generates response using local LLM, context, and chat history."""
        try:
            # Determine system message based on context and config
            system_msg = self.cfg.model.SYSTEM_MESSAGE or "You are a helpful AI assistant." # Default to config or generic
            is_domain_check_off = not self.cfg.retrieval.PERFORM_DOMAIN_CHECK
            is_context_empty = not context or not context.strip()

            if is_domain_check_off and is_context_empty:
                # Use a purely generic message when domain check is OFF and no context found
                system_msg = "You are a helpful AI assistant."
                self.logger.info("Using generic system message for local LLM (no context, domain check OFF).")
            # else: Use the system_msg loaded from config (which might be domain-specific)

            history_str = self._format_chat_history_for_prompt(chat_history)
            context_part = f"Context:\n{context}\n\n" if context else "Context: [No relevant information found in documents]\n\n"
            prompt = f"{system_msg}\n\n{history_str}\n\n{context_part}User:\n{query}"
            # --- ADD THIS LOGGING ---
            self.logger.info("="*20 + " FINAL LOCAL LLM PROMPT " + "="*20)
            self.logger.info(f"System Message Part:\n{system_msg_base}") # Check if new message loaded
            self.logger.info(f"\nHistory Part:\n{history_str}") # Check history formatting
            self.logger.info(f"\nContext Part (First 500 chars):\n{context_part[:500]}...") # Check retrieved context relevance
            self.logger.info(f"\nFinal User Query Part:\nUser:\n{query}")
            self.logger.info("="*60)
            # --- END LOGGING ---

            response_content = self.local_llm.invoke(prompt)
            return str(response_content).strip() if response_content is not None else "Local LLM did not return a response."
        except Exception as e:
            logger.error(f"Local LLM invocation failed: {e}", exc_info=True)
            return f"Local LLM encountered an error: {str(e)}" # Return error message

    # === MODIFIED: call_deepseek_api ===
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry_error_callback=lambda retry_state: logger.warning(f"Retrying DeepSeek API call after error: {retry_state.outcome.exception()}")
    )
    def call_deepseek_api(self, query: str, context: str, chat_history: Optional[List[Dict]] = None) -> Optional[str]:
        """Calls DeepSeek API with context and chat history."""
        if not self.cfg.security.DEEPSEEK_API_KEY:
            self.logger.info("DeepSeek API key not configured. Skipping API call.")
            return None

        # --- Caching Check (Modified Key - Optional, consider if API calls are expensive/deterministic) ---
        # history_str_api = json.dumps(chat_history) if chat_history else ""
        # api_cache_key_content = f"api_{query}_{self._get_cache_key(context)}_{history_str_api}"
        # api_cache_key = self._get_cache_key(api_cache_key_content)
        # cached = self._load_from_cache(api_cache_key)
        # if cached:
        #    self.logger.info("Returning CACHED DeepSeek API response")
        #    return cached.get("response")
        # --- End Caching Check ---

        try:
            headers = {
                "Authorization": f"Bearer {self.cfg.security.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }

            system_msg_base = self.cfg.model.SYSTEM_MESSAGE or "You are a helpful industrial automation expert. Use the provided context and conversation history."
            system_msg = f"{system_msg_base}\n\nContext:\n{context}" if context else system_msg_base

            # Format history for API
            api_history_messages = self._format_chat_history_for_api(chat_history)

            # Construct messages list: System + History + Current User Query
            messages = [{"role": "system", "content": system_msg}] + api_history_messages + [{"role": "user", "content": query}]

            payload = {
                "model": "deepseek-chat", # Use appropriate model name
                "messages": messages,
                "temperature": self.cfg.model.LLM_TEMPERATURE,
                "max_tokens": self.cfg.model.MAX_TOKENS,
                "top_p": self.cfg.model.TOP_P,
                "frequency_penalty": self.cfg.model.FREQUENCY_PENALTY,
            }

            self.logger.debug(f"DeepSeek API Payload Messages (Preview): {json.dumps(messages, indent=2)[:500]}...")

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.cfg.security.API_TIMEOUT
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result_data = response.json()

            if result_data and "choices" in result_data and result_data["choices"]:
                api_result = result_data["choices"][0].get("message", {}).get("content")
                if api_result:
                    # Optional: Save to cache if enabled
                    # self._save_to_cache(api_cache_key, {"response": api_result.strip()})
                    return api_result.strip()
                else:
                     self.logger.warning("API response missing content in choices.")
                     return None
            else:
                self.logger.error(f"API response missing expected structure: {result_data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API request failed: {e}", exc_info=True)
            return None # Return None on failure
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"DeepSeek API response parsing failed: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"DeepSeek API call failed unexpectedly: {str(e)}", exc_info=True)
            return None

    def validate_technical_response(self, response: str) -> bool:
        """Validates if the response seems technically relevant (basic keyword check)."""
        if not response: return False
        # Keep your existing regex patterns
        technical_indicators = [
            r"\b(Experion\s+HS|Experion|CCC\s+(?:Connect|Inside)|ControlEdge|Honeywell\s+(?:Controller(?:s)?|Indicator(?:s)?|Set[-\s]?Point\s*Programmer(?:s)?|Universal\s*Digital\s*Controller|UDC\s*\d+|DC\s*\d+|DCP\s*\d+))\b",
            r"\b(SCADA|HMI|DCS|RTU|PLC|OPC\s*(?:UA|DA)|Modbus|PROFIBUS|EtherCAT)\b",
            # ... (keep all other patterns) ...
            r"\b(throughput\s+increase|recovery\s*rate|MTBF|MTTR)\b"
        ]
        # Use re.IGNORECASE for case-insensitivity
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in technical_indicators)

    def merge_responses(self, local: str, api: Optional[str]) -> str:
        """Merges responses from local LLM and optional API verification."""
        # Prioritize API response if available and non-empty
        if api:
            self.logger.debug("Using API response as primary.")
            return api # Simple strategy: just return API if it exists
        else:
            self.logger.debug("Using Local LLM response (API not available or failed).")
            return local
        # --- Old merge logic (can uncomment if needed) ---
        # if not api: return local
        # # Basic check if responses are very similar (can be improved)
        # if local.strip().lower() == api.strip().lower():
        #     return local # Return one if identical
        # # Simple concatenation if different (consider more sophisticated merging)
        # return f"Local Response:\n{local}\n\nAPI Verification:\n{api}"
        # --- End Old merge logic ---


    def _format_response(self, response: str, source: str, context: Optional[str], error: bool = False) -> Dict:
        """Formats the final response dictionary."""
        model_used = self.cfg.model.OLLAMA_MODEL # Default
        # Add logic to determine model if API was the source (depends on merge_responses logic)
        if source == "merged" and self.cfg.security.DEEPSEEK_API_KEY: # Assuming merged implies API was used if key exists
             model_used = "deepseek-chat" # Or determine more accurately

        return {
            # Use "response" key consistent with what app.py currently expects from pipeline
            "response": response,
            "source": source,
            "context": context if context else "", # Ensure context is string
            "model": model_used,
            "timestamp": datetime.now().isoformat(),
            "error": error # Include error flag
        }

# === Keep direct execution block for testing (if desired) ===
if __name__ == "__main__":
    # Note: Direct testing here won't simulate chat history easily.
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__) # Re-get logger instance after basicConfig
    logger.setLevel(logging.DEBUG)

    test_query = sys.argv[1] if len(sys.argv) > 1 else "Compare SCADA from Honeywell and SAP."

    test_pipeline = None
    try:
        print(f"\n--- Testing Pipeline with Query: '{test_query}' ---")
        test_pipeline = IndustrialAutomationPipeline()
        # --- Test a simple query without history ---
        result = test_pipeline.generate_response(test_query, chat_history=None)
        print("\n>>> Result (No History):")
        print(json.dumps(result, indent=2))
        print("-" * 60)

        # --- Simulate a small history ---
        sim_history = [
             {"role": "user", "content": "What is SCADA?", "timestamp": "..."},
             {"role": "assistant", "content": "SCADA stands for Supervisory Control and Data Acquisition...", "timestamp": "...", "source": "...", "model": "..."},
        ]
        follow_up_query = "How is it different from DCS?"
        print(f"\n--- Testing Pipeline with Follow-up: '{follow_up_query}' ---")
        result_hist = test_pipeline.generate_response(follow_up_query, chat_history=sim_history)
        print("\n>>> Result (With History):")
        print(json.dumps(result_hist, indent=2))
        print("-" * 60)


    except Exception as main_e:
        logger.critical(f"Error during direct test execution: {main_e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if test_pipeline and hasattr(test_pipeline.retriever, 'close'):
            try:
                test_pipeline.retriever.close()
                logger.info("Closed Weaviate client for test instance.")
            except Exception as close_e:
                 logger.error(f"Error closing retriever client: {close_e}")
        print("\n--- Test finished ---")
