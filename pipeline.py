# pipeline.py (Corrected for Chat History)

import re
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any
# Strategy imports for response merging
from llm_merge_strategies import (
    MergeStrategy,
    ApiPriorityStrategy,
    ConcatStrategy,
    LocalOnlyStrategy,
)
from sklearn.metrics.pairwise import cosine_similarity
from llm_providers import get_provider, LLMProvider
from dotenv import load_dotenv
from config import cfg



import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
import os
import numpy as np
import socket
import logging


load_dotenv()
# module-level self/api_provider removed — it belongs in the pipeline __init__
  
    
# Initialize logger
logger = logging.getLogger(__name__)
PIPELINE_INSTANCE = None
_pipeline_initializing = False


## Core imports and dummy‐fallbacks…
try:
    from langchain_ollama import OllamaLLM
    from config import cfg # Use 'cfg' directly if it's the loaded instance
    from retriever import TechnicalRetriever
    from llm_merge_strategies import ApiPriorityStrategy, ConcatStrategy, LocalOnlyStrategy, MergeStrategy
    
    # Try to import CentroidManager
    try:
        from centroid_manager import CentroidManager
        centroid_manager = CentroidManager()
        centroid_available = True
        logger.info("CentroidManager initialized successfully")
    except ImportError:
        logger.warning("CentroidManager not available, centroid features disabled")
        centroid_manager = None
        centroid_available = False
        
    imports_ok = True
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import core pipeline dependencies: {e}", exc_info=True)
    imports_ok = False
    # Define dummy classes/objects to prevent immediate crash if possible, but log error
    #class OllamaLLM: pass
    #class TechnicalRetriever: pass
    class cfg: # Dummy config
        class model: OLLAMA_MODEL="dummy"; LLM_TEMPERATURE=0.7; MAX_TOKENS=512; SYSTEM_MESSAGE=""
        class security: SANITIZE_INPUT=True; DEEPSEEK_API_KEY=""; API_TIMEOUT=20; CACHE_ENABLED=False
        class retrieval: PERFORM_DOMAIN_CHECK=False; DOMAIN_SIMILARITY_THRESHOLD=0.6; SPARSE_RELEVANCE_THRESHOLD=0.1; FUSED_RELEVANCE_THRESHOLD=0.4; SEMANTIC_WEIGHT=0.7; SPARSE_WEIGHT=0.3
        class paths: DOMAIN_CENTROID_PATH="./dummy_centroid.npy"
        class env: merged_keywords=[]
    centroid_manager = None

## Configure logging (after imports)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        if not imports_ok:
            self.logger.critical("Pipeline cannot initialize due to missing imports.")
            raise ImportError("Core pipeline dependencies failed to import.")

        self.cfg = cfg

        try:
            self.local_llm = OllamaLLM(
                model=self.cfg.model.OLLAMA_MODEL,
                temperature=self.cfg.model.LLM_TEMPERATURE,
            )

            self.retriever = TechnicalRetriever()

            if hasattr(self.retriever, "weaviate_client") and self.retriever.weaviate_client:
                try:
                    self.retriever.weaviate_client.connect()
                    self.logger.info("Weaviate client successfully (re)connected.")
                except Exception as e:
                    self.logger.critical(f"Failed to reconnect Weaviate client: {e}", exc_info=True)
                    raise RuntimeError("Pipeline init failed: could not reconnect Weaviate client.")

            self.embeddings = self.retriever.embeddings

        except Exception as init_e:
            self.logger.critical(f"Failed to initialize LLM or Retriever: {init_e}", exc_info=True)
            raise

        self.domain_keywords_set = set(getattr(self.cfg.env, 'merged_keywords', []))
        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.domain_centroid = self._load_domain_centroid()
        if self.domain_centroid is None and self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
            self.logger.warning("Domain centroid not loaded. Semantic relevance check will be skipped.")

        if not self.domain_keywords_set and self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
            self.logger.warning("Domain keywords list is EMPTY during pipeline initialization!")
        else:
            self.logger.info(f"Pipeline initialized with {len(self.domain_keywords_set)} domain keywords.")

        # Validators
        self._validators = []
        if self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
            self._validators.append(self._validate_semantic)
        if self.cfg.retrieval.PERFORM_TECHNICAL_VALIDATION:
            self._validators.append(self._validate_technical)

        # Merge strategy
        strat = getattr(self.cfg.model, "MERGE_STRATEGY", "api_first").lower()
        if strat == "concat":
            self.merge_strategy: MergeStrategy = ConcatStrategy()
        elif strat == "local_only":
            self.merge_strategy = LocalOnlyStrategy()
        else:
            self.merge_strategy = ApiPriorityStrategy()

        # External API provider
        self.api_provider: Optional[LLMProvider] = get_provider(
            self.cfg,
            self._format_chat_history_for_api
        )
        self._connected_host = self.cfg.retrieval.WEAVIATE_HOST
        self._connected_port = self.cfg.retrieval.WEAVIATE_HTTP_PORT


    def _load_domain_centroid(self) -> Optional[np.ndarray]:
        """Loads the pre-calculated centroid vector for the configured collection."""
        try:
            from centroid_manager import CentroidManager

            # 1) Grab the collection name from config
            collection_name = getattr(self.cfg.retrieval, "COLLECTION_NAME", None)

            # 2) Use the dedicated CENTROID_DIR, not DOMAIN_CENTROID_PATH’s folder
            base_dir = getattr(self.cfg.paths, "CENTROID_DIR", None)
            if not base_dir:
                # fallback in case CENTROID_DIR isn’t set
                base_dir = os.path.dirname(self.cfg.paths.DOMAIN_CENTROID_PATH) or "."

            # 3) Instantiate the manager with per-collection behavior
            cm = CentroidManager(
                collection_name=collection_name,
                base_path=base_dir
            )

            centroid = cm.get_centroid()

            if centroid is not None:
                logger.info(
                    f"Loaded centroid for collection '{collection_name}' "
                    f"from {cm.centroid_path} with shape {centroid.shape}"
                )
                # reshape for cosine_similarity
                return centroid.reshape(1, -1)
            else:
                logger.error(
                    f"Centroid file not found for collection '{collection_name}' "
                    f"at {cm.centroid_path}. Run calculate_centroid.py."
                )
                return None

        except Exception as e:
            logger.error(f"Failed to load domain centroid: {e}", exc_info=True)
            return None


        
    def _validate_semantic(self, text: str) -> bool:
        # (optional) check domain centroid here if you ever want to re-enable it
        return True

    def _validate_technical(self, text: str) -> bool:
            # your keyword-based check from before
            terms = [k.lower() for k in self.cfg.env.DOMAIN_KEYWORDS]
            return any(term in text.lower() for term in terms)

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


    def _calculate_semantic_score(self, query: str) -> Tuple[float, Dict]:
        """Calculates cosine similarity between query embedding and domain centroid."""
        if self.domain_centroid is None:
            logger.warning("Cannot calculate semantic score, domain centroid not loaded.")
            return 0.0, {}
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            centroid_feedback = centroid_manager.query_insight(query_vector.flatten())
            if centroid_feedback and "similarity" in centroid_feedback:
                similarity = centroid_feedback["similarity"]
            else:
                similarity = cosine_similarity(query_vector, self.domain_centroid)[0][0]
            similarity = max(0.0, min(1.0, float(similarity)))
            return similarity, centroid_feedback
        except Exception as e:
            logger.error(f"Error calculating semantic score: {e}", exc_info=True)
            return 0.0, {}    


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

    def _check_domain_relevance_hybrid(self, query: str, query_terms: List[str]) -> Tuple[bool, Dict]:
        """Performs hybrid relevance check using semantic and sparse scores."""
        semantic_score, centroid_feedback = self._calculate_semantic_score(query)
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
        return is_relevant,centroid_feedback

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
        History-aware cache → optional domain check → retrieve → generate (local+API)
        → optional validation → format → cache.
        """
        from datetime import datetime
        import json

        self.logger.info(
            f"Generating response (max_history_turns={self.cfg.pipeline.max_history_turns}, "
            f"history len={len(chat_history) if chat_history else 0})"
        )

        # ── CACHE KEY ──
        max_msgs = self.cfg.pipeline.max_history_turns * 2
        if chat_history and len(chat_history) > max_msgs:
            hist_cache = chat_history[-max_msgs - 1:-1]
        else:
            hist_cache = chat_history[:-1] if chat_history else None
        hist_key = self._format_chat_history_for_api(hist_cache) or ""
        cache_key = self._get_cache_key(f"{query}_{json.dumps(hist_key)}")
        if (cached := self._load_from_cache(cache_key)):
            self.logger.info("Returning cached response")
            cached.setdefault('response','')
            cached.setdefault('source','cache')
            cached.setdefault('model','cache')
            cached.setdefault('context','')
            cached.setdefault('error',False)
            cached['timestamp'] = datetime.now().isoformat()
            return cached

        centroid_feedback = {}
        try:
            # ── SANITIZE ──
            sanitized = self._sanitize_input(query) if self.cfg.security.SANITIZE_INPUT else query
            if self.cfg.security.SANITIZE_INPUT and (not sanitized or len(sanitized) < 3):
                return self._format_response(
                    response="Query too short or invalid after sanitization",
                    source="validation", context=None, error=True
                )

            terms = [t for t in sanitized.lower().split() if len(t) > 1]

            # ── DOMAIN CHECK ──
            if self.cfg.retrieval.PERFORM_DOMAIN_CHECK:
                ok, centroid_feedback = self._check_domain_relevance_hybrid(sanitized, terms)
                if not ok:
                    return self._format_response(
                        response="Query appears outside industrial automation scope.",
                        source="domain_check", context=None, error=False
                    )
                self.logger.info("Domain check passed")
            else:
                self.logger.info("Domain check skipped")

            # ── RETRIEVE CONTEXT ──
            retrieval_q = sanitized
            if self.cfg.retrieval.retrieve_with_history and hist_cache:
                last_user = next((m["content"] for m in reversed(hist_cache) if m.get("role")=="user"), "")
                if last_user:
                    retrieval_q = f"Previous: {last_user}\nCurrent: {sanitized}"
            self.logger.info(f"Retrieving context for: '{retrieval_q[:100]}…'")
            context = self.retriever.get_context(retrieval_q) or ""
            if not context:
                self.logger.warning("No context found")

            # 1) Local LLM response always as dict
            local_resp = {"response": self._generate_local_response(sanitized, context, chat_history)}

            # 2) External provider call (None if not configured)
            prov = self.cfg.model.EXTERNAL_API_PROVIDER.lower()
            api_raw = self.api_provider.call(sanitized, context, chat_history) if self.api_provider else None
            api_resp = api_raw if isinstance(api_raw, dict) else {}


            # 3) Warn if user wanted a provider but got nothing back
            if prov != "none" and not api_raw:
                self.logger.warning(f"Provider '{prov}' configured but returned no response or is unavailable")

            # 4) Always merge dicts
            api_resp: Dict = api_raw if isinstance(api_raw, dict) else {}
            merged = self.merge_strategy.merge(local_resp, api_resp)
            text = merged.get("response", "")

            # ── VALIDATION ──
            is_valid = True
            source = prov if api_resp else "local_llm"
            if self.cfg.retrieval.PERFORM_TECHNICAL_VALIDATION:
                for vfn in self._validators:
                    if not vfn(text):
                        self.logger.info(f"Validation failed: {vfn.__name__}")
                        merged = {"response":"Please clarify or ask a specific technical topic."}
                        source = "validation_failed"
                        is_valid = False
                        break
            else:
                self.logger.info("Technical validation skipped")

            # ── FORMAT ──
            final = self._format_response(
                response=merged,
                source=source,
                context=context if is_valid else "",
                error=False
            )
            final["centroid_feedback"] = centroid_feedback

            # ── CACHE if valid ──
            if is_valid:
                self._save_to_cache(cache_key, final)
            else:
                self.logger.info("Not caching invalidated response")

            return final

        except Exception as e:
            self.logger.exception(f"Error in generate_response: {e}")
            return self._format_response(
                response="Sorry, an internal error occurred.",
                source="pipeline_error", context=None, error=True
            )


    # === MODIFIED: _generate_local_response ===
    def _generate_local_response(self, query: str, context: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Generates response using local LLM, context, and chat history."""
        try:
            system_msg_base = self.cfg.model.SYSTEM_MESSAGE or "You are a helpful industrial automation expert. Use the following context and conversation history to answer the question."

            # Format history string
            history_str = self._format_chat_history_for_prompt(chat_history)

            # Construct prompt with history
            context_part = f"Context:\n{context}\n\n" if context else ""
            # Ensure clear separation and final instruction
            prompt = f"{system_msg_base}\n\n{history_str}\n\n{context_part}User:\n{query}"

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
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_deepseek_api(self, query: str, context: Optional[str], chat_history: Optional[List[Dict]]) -> Optional[Dict]:
        """
        Calls the DeepSeek Chat Completion API, adapting the system prompt
        based on domain check settings and context availability.
        """
        provider_name = "deepseek"
        api_key = self.cfg.security.DEEPSEEK_API_KEY
        timeout = self.cfg.security.API_TIMEOUT

        if not api_key:
            self.logger.warning(f"{provider_name.capitalize()} API key not configured. Skipping API call.")
            return None

        api_url = "https://api.deepseek.com/chat/completions" # Verify DeepSeek endpoint URL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # --- Determine System Prompt ---
        default_system_message = self.cfg.model.SYSTEM_MESSAGE or "You are a helpful AI assistant."
        system_prompt_content = default_system_message
        is_domain_check_off = not self.cfg.retrieval.PERFORM_DOMAIN_CHECK
        is_context_empty = not context or not context.strip()

        if is_domain_check_off and is_context_empty:
            # Use a generic prompt if domain check is OFF and no context was retrieved
            system_prompt_content = "You are a helpful AI assistant. Answer the user's query based on your general knowledge."
            self.logger.info(f"Using GENERIC system prompt for {provider_name} API (Domain Check OFF, No Context).")
        else:
            # Use the configured (potentially domain-specific) system prompt
             self.logger.info(f"Using CONFIGURED system prompt for {provider_name} API.")
        # --- End System Prompt Determination ---

        # Format history for API
        api_history = self._format_chat_history_for_api(chat_history)

        # Construct messages payload
        messages = [{"role": "system", "content": system_prompt_content}]
        messages.extend(api_history) # Add formatted history
        # Add context if available (and domain check wasn't off with empty context)
        if context and not (is_domain_check_off and is_context_empty):
             messages.append({"role": "system", "content": f"Use the following context to answer:\n{context}"})
        # Add the current user query
        messages.append({"role": "user", "content": query})

        # Pick override → per-provider default → (never) literal fallback
        default_model =self.cfg.model.EXTERNAL_API_MODEL_DEFAULTS.get("deepseek")
        model_name    = self.cfg.model.EXTERNAL_API_MODEL_NAME or default_model
        payload = {
            "model":     model_name,
             "messages":  messages,
             "temperature": self.cfg.model.LLM_TEMPERATURE,
             "max_tokens":  self.cfg.model.MAX_TOKENS,
             "top_p":       self.cfg.model.TOP_P,
             "frequency_penalty": self.cfg.model.FREQUENCY_PENALTY,
             "stream":     False
         }

        self.logger.info(f"Calling {provider_name.capitalize()} API: {api_url} with model {payload['model']}")
        self.logger.debug(f"API Payload (excluding messages): { {k:v for k,v in payload.items() if k != 'messages'} }")
        self.logger.debug(f"API Messages count: {len(messages)}")

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            api_result = response.json()
            # Log raw response for debugging if needed
            self.logger.debug(f"{provider_name.capitalize()} API Raw Response: {api_result}")
         

            # Extract the response text (adjust based on actual DeepSeek API structure)
            response_text = api_result.get("choices", [{}])[0].get("message", {}).get("content", "")
            model_used = api_result.get("model", payload['model'])

            if not response_text:
                 self.logger.warning(f"{provider_name.capitalize()} API returned an empty response.")
                 return None # Or return an error dict

            self.logger.info(f"{provider_name.capitalize()} API call successful. Response length: {len(response_text)}")
            return {
                "response": response_text.strip(),
                "source": provider_name,
                "model": model_used,
                "error": False
            }

        except requests.exceptions.Timeout:
            self.logger.error(f"{provider_name.capitalize()} API request timed out after {timeout} seconds.")
            return {"response": f"Error: API request timed out ({provider_name}).", "source": provider_name, "model": payload['model'], "error": True}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"{provider_name.capitalize()} API request failed: {e}", exc_info=True)
            error_detail = str(e)
            # Try to get more detail from response if available
            if e.response is not None:
                try:
                    error_body = e.response.json()
                    error_detail = error_body.get("error", {}).get("message", str(e))
                except json.JSONDecodeError:
                    error_detail = e.response.text[:200] # Use raw text if not JSON

            return {"response": f"Error: API request failed ({provider_name}): {error_detail}", "source": provider_name, "model": payload['model'], "error": True}
        except Exception as e:
            self.logger.error(f"Unexpected error during {provider_name} API call: {e}", exc_info=True)
            return {"response": f"Error: Unexpected issue during API call ({provider_name}).", "source": provider_name, "model": payload['model'], "error": True}
   

        # un comment and place properly if needed
        # --- Caching Check (Modified Key - Optional, consider if API calls are expensive/deterministic) ---
        # history_str_api = json.dumps(chat_history) if chat_history else ""
        # api_cache_key_content = f"api_{query}_{self._get_cache_key(context)}_{history_str_api}"
        # api_cache_key = self._get_cache_key(api_cache_key_content)
        # cached = self._load_from_cache(api_cache_key)
        # if cached:
        #    self.logger.info("Returning CACHED DeepSeek API response")
        #    return cached.get("response")
        # --- End Caching Check ---
       
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

    def merge_responses(self, local: str, api: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges responses from local LLM and optional API verification."""
        if isinstance(api, dict):
            self.logger.debug("Using API response as primary.")
            return api

        self.logger.debug("Using Local LLM response (API not available or failed).")
        # wrap local string into a dict so .get() works
        return {"response": local}

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


def is_pipeline_valid(p):
    """
    Return True if `p` has a live retriever.weaviate_client.
    """
    return (
        p is not None
        and hasattr(p, "retriever")
        and getattr(p.retriever, "weaviate_client", None) is not None
        and p.retriever.weaviate_client.is_ready()
    )


def init_pipeline_once(force: bool = False):
    """
    Lazily (re)initialize the global PIPELINE_INSTANCE only when needed.
    Rebuilds if force=True, never built one yet, or host/port changed.
    """
    global PIPELINE_INSTANCE

    needs_rebuild = force or PIPELINE_INSTANCE is None

    if PIPELINE_INSTANCE:
        prev_host = getattr(PIPELINE_INSTANCE, "_connected_host", None)
        prev_port = getattr(PIPELINE_INSTANCE, "_connected_port", None)
        curr_host = cfg.retrieval.WEAVIATE_HOST
        curr_port = cfg.retrieval.WEAVIATE_HTTP_PORT

        if (prev_host, prev_port) != (curr_host, curr_port):
            logger.info(
                "[Pipeline] Weaviate connection settings changed: "
                f"{prev_host}:{prev_port} -> {curr_host}:{curr_port}; rebuilding."
            )
            needs_rebuild = True

    if not needs_rebuild:
        return PIPELINE_INSTANCE

    # Tear down old client if exists
    if PIPELINE_INSTANCE and hasattr(PIPELINE_INSTANCE, "retriever"):
        try:
            PIPELINE_INSTANCE.retriever.close()
            logger.info("[Pipeline] Closed old retriever.")
        except Exception as e:
            logger.warning(f"[Pipeline] Failed to close old retriever: {e}")

    # Build new
    logger.info("[Pipeline] Initializing new UnifiedPipeline…")
    PIPELINE_INSTANCE = UnifiedPipeline()

    # Remember connection info
    PIPELINE_INSTANCE._connected_host = cfg.retrieval.WEAVIATE_HOST
    PIPELINE_INSTANCE._connected_port = cfg.retrieval.WEAVIATE_HTTP_PORT

    logger.info(
        "[Pipeline] New pipeline connected to "
        f"{PIPELINE_INSTANCE._connected_host}:"
        f"{PIPELINE_INSTANCE._connected_port}."
    )

    return PIPELINE_INSTANCE



def initialize_pipeline(app_context=None, force=False):
    """
    Initializes or re-initializes the global pipeline object.
    Skips if already valid or initialization is running.
    """
    global _pipeline_initializing, PIPELINE_INSTANCE

    if _pipeline_initializing:
        logger.info("Pipeline init already in progress. Skipping.")
        return

    _pipeline_initializing = True
    try:
        # Optionally use the Flask app context if provided
        if app_context:
            with app_context:
                _inner_init(force)
        else:
            _inner_init(force)
    finally:
        _pipeline_initializing = False


def _inner_init(force):
    """
    Helper to do the actual initialization logic under whichever context.
    """
    global PIPELINE_INSTANCE

    host = cfg.retrieval.WEAVIATE_HOST
    port = cfg.retrieval.WEAVIATE_HTTP_PORT

    # Quick reachability check
    try:
        socket.create_connection((host, port), timeout=2).close()
        logger.info(f"Weaviate reachable at {host}:{port}")
    except Exception as e:
        logger.warning(f"Weaviate not reachable at {host}:{port}: {e}")
        PIPELINE_INSTANCE = None
        return

    # Rebuild if invalid or forced
    if not is_pipeline_valid(PIPELINE_INSTANCE) or force:
        logger.info("Rebuilding pipeline (invalid or force requested)")
        PIPELINE_INSTANCE = init_pipeline_once(force=force)
    else:
        logger.info("Existing pipeline is valid; skipping rebuild")


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
        test_pipeline = UnifiedPipeline()
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