# llm_merge_strategies.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class MergeStrategy(ABC):
    @abstractmethod
    def merge(self, local: Dict[str, Any], api: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a merged response dict with at least a 'response' key."""
        ...

class ApiPriorityStrategy(MergeStrategy):
    def merge(self, local: Dict[str, Any], api: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # If API returned a dict with a non-empty 'response', use it directly
        if api and api.get("response"):
            return api
        return local

class ConcatStrategy(MergeStrategy):
    def merge(self, local: Dict[str, Any], api: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        l = local.get("response", "").strip()
        a = (api or {}).get("response", "").strip()
        if not a or a.lower() == l.lower():
            return {"response": l}
        return {"response": f"Local Response:\n{l}\n\nAPI Response:\n{a}"}

class LocalOnlyStrategy(MergeStrategy):
    def merge(self, local: Dict[str, Any], api: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return local
