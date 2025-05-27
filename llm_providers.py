import os
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, List, Dict, Any

class LLMProvider:
    def call(
        self,
        query: str,
        context: Optional[str],
        history: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Abstract interface for LLM providers"""
        raise NotImplementedError()

class GenericChatProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        api_url: str,
        api_key: str,
        default_system: str,
        perform_domain_check: bool,
        format_history_fn,
        model_name: str,
        timeout: int,
        extra_params: Dict[str, Any] = None
    ):
        self.name = name
        self.api_url = api_url
        self.api_key = api_key
        self.default_system = default_system
        self.domain_check = perform_domain_check
        self.format_history = format_history_fn
        self.model_name = model_name
        self.timeout = timeout
        self.extra = extra_params or {}
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call(
        self,
        query: str,
        context: Optional[str],
        chat_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        prov = self.name
        if not self.api_key:
            self.logger.warning(f"{prov} key missing, skipping API call.")
            return None

        # Select system prompt
        empty_ctx = not context or not context.strip()
        if not self.domain_check and empty_ctx:
            sys_msg = "You are a helpful AI assistant. Answer from general knowledge."
            self.logger.info(f"Using GENERIC system prompt for {prov}.")
        else:
            sys_msg = self.default_system
            self.logger.info(f"Using CONFIGURED system prompt for {prov}.")

        # Build message list
        history_msgs = self.format_history(chat_history) or []
        messages = [{"role": "system", "content": sys_msg}] + history_msgs
        if context and not (not self.domain_check and empty_ctx):
            messages.append({"role": "system", "content": f"Use context:\n{context}"})
        messages.append({"role": "user", "content": query})

        # Construct payload differently for DeepSeek vs others
        if prov == "deepseek":
            payload = {
                "model":            self.model_name,
                "messages":         messages,
                "frequency_penalty": self.extra.get("frequency_penalty", 0),
                "presence_penalty":  0,
                "max_tokens":        self.extra.get("max_tokens", 2048),
                "temperature":       self.extra.get("temperature", 1),
                "top_p":             self.extra.get("top_p", 1),
                "response_format":   {"type": "text"},
                "stop":              None,
                "stream":            False,
                "stream_options":    None,
                "tools":             None,
                "tool_choice":       "none",
                "logprobs":          False,
                "top_logprobs":      None
            }
        else:
            payload = {
                "model":    self.model_name,
                "messages": messages,
                **self.extra,
                "stream":   False
            }

        headers = {
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            if not text:
                return None
            return {
                "response": text.strip(),
                "source":   prov,
                "model":    data.get("model", self.model_name),
                "error":    False
            }
        except requests.exceptions.Timeout:
            msg = f"Error: API request timed out ({prov})."
            return {"response": msg, "source": prov, "model": self.model_name, "error": True}
        except Exception as e:
            self.logger.error(f"{prov} API error: {e}", exc_info=True)
            return {"response": f"Error: API failure ({prov}): {e}", "source": prov, "model": self.model_name, "error": True}


def get_provider(cfg, format_history_fn) -> Optional[LLMProvider]:
    name = cfg.model.EXTERNAL_API_PROVIDER.lower()
    if name == "none":
        return None

    url = cfg.model.PROVIDER_URLS[name]
    key = os.getenv(f"{name.upper()}_API_KEY", "")

    default_model = cfg.model.EXTERNAL_API_MODEL_DEFAULTS.get(name)
    model_name    = cfg.model.EXTERNAL_API_MODEL_NAME or default_model or name

    extra = {
        "temperature":       cfg.model.LLM_TEMPERATURE,
        "max_tokens":        cfg.model.MAX_TOKENS,
        "top_p":             cfg.model.TOP_P,
        "frequency_penalty": cfg.model.FREQUENCY_PENALTY
    }

    return GenericChatProvider(
        name=name,
        api_url=url,
        api_key=key,
        default_system=cfg.model.SYSTEM_MESSAGE,
        perform_domain_check=cfg.retrieval.PERFORM_DOMAIN_CHECK,
        format_history_fn=format_history_fn,
        model_name=model_name,
        timeout=cfg.security.API_TIMEOUT,
        extra_params=extra
    )
