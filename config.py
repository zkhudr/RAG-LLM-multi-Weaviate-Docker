# config.py (Updated for Multiple Secure API Keys & Generic Provider)

import os
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing import List, Dict, Any, Optional, Literal # Added Literal
from dotenv import load_dotenv
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env variables into os.environ FIRST
load_dotenv()

# Configuration File Path
CONFIG_YAML_PATH = Path("./config_settings.yaml").resolve()

# --- Flow Style for YAML Lists ---
class FlowStyleList(list): pass
def flow_style_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Use SafeDumper and add representer conditionally
dumper_to_use = yaml.SafeDumper
if 'FlowStyleList' in globals() and 'flow_style_list_representer' in globals():
    class ConfigDumper(yaml.SafeDumper): pass
    ConfigDumper.add_representer(FlowStyleList, flow_style_list_representer)
    dumper_to_use = ConfigDumper
# --- End Flow Style ---

def save_yaml_config(data: Dict[str, Any], path: Path):
    """Atomically save validated config data dictionary to YAML file."""
    temp_path = path.with_suffix(".yaml.tmp")
    try:
        dump_data = data.copy()
        if FlowStyleList and 'env' in dump_data:
             env_section = dump_data['env']
             for key in ['DOMAIN_KEYWORDS', 'AUTO_DOMAIN_KEYWORDS', 'USER_ADDED_KEYWORDS']:
                 if key in env_section and isinstance(env_section[key], list):
                     env_section[key] = FlowStyleList(env_section[key])

        logger.debug(f"Dumping config data to temporary file: {temp_path}")
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(dump_data, f, default_flow_style=None, sort_keys=False, Dumper=dumper_to_use, indent=2)
        logger.debug("YAML dump successful.")

        logger.debug(f"Attempting atomic replace: {temp_path} -> {path}")
        try: os.replace(temp_path, path)
        except OSError as replace_error:
            logger.warning(f"os.replace failed ({replace_error}), attempting shutil.move...")
            shutil.move(str(temp_path), str(path))

        logger.info(f"Successfully saved configuration to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save YAML configuration to {path}: {e}", exc_info=True)
        if temp_path.exists(): # Clean up temp file on error
            try: temp_path.unlink()
            except OSError as e_rm: logger.error(f"Failed to remove temp file {temp_path}: {e_rm}")
        raise # Re-raise the original error

# === Pydantic Models ===

class SecurityConfig(BaseModel):
    SANITIZE_INPUT: bool = True
    RATE_LIMIT: int = 10
    API_TIMEOUT: int = 30
    CACHE_ENABLED: bool = False

    # API Keys - Loaded ONLY from environment variables, default to empty string
    DEEPSEEK_API_KEY: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""), exclude=True)
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""), exclude=True)
    ANTHROPIC_API_KEY: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""), exclude=True)
    COHERE_API_KEY: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""), exclude=True)
    # Add other keys here following the pattern

    # Note: exclude=True prevents these fields from being included in model_dump()
    # which prevents them from being saved back to YAML.

class RetrievalConfig(BaseModel):
    COLLECTION_NAME: str = "industrial_tech"
    K_VALUE: int = 6
    SCORE_THRESHOLD: float = 0.6
    LAMBDA_MULT: float = 0.6
    SEARCH_TYPE: str = "mmr"
    DOMAIN_SIMILARITY_THRESHOLD: float = 0.65
    SPARSE_RELEVANCE_THRESHOLD: float = 0.15
    FUSED_RELEVANCE_THRESHOLD: float = 0.45
    SEMANTIC_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    PERFORM_DOMAIN_CHECK: bool = True

    @field_validator('SEARCH_TYPE')
    def validate_search_type(cls, value):
        allowed = {'mmr', 'similarity', 'similarity_score_threshold'}
        val_lower = value.lower()
        if val_lower not in allowed:
            raise ValueError(f"SEARCH_TYPE must be one of {allowed}, got '{value}'")
        return val_lower

# Literal type for provider choices
ApiProvider = Literal["deepseek", "openai", "anthropic", "cohere", "none"]

class ModelConfig(BaseModel):
    LLM_TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 1536
    OLLAMA_MODEL: str = "deepseek-coder:6.7b-instruct-q5_K_M" # Primary local model
    EMBEDDING_MODEL: str = "nomic-embed-text"
    TOP_P: float = 0.9
    FREQUENCY_PENALTY: float = 0.1
    SYSTEM_MESSAGE: str = Field(default="You are an expert AI assistant...") # Keep enhanced message

    # --- NEW: Select External API Provider ---
    EXTERNAL_API_PROVIDER: ApiProvider = "deepseek" # Default provider
    EXTERNAL_API_MODEL_NAME: Optional[str] = None # Optional: Override default model name for provider

    @field_validator('EXTERNAL_API_PROVIDER')
    def validate_provider(cls, value):
        # Simple lowercase validation, Literal handles allowed values
        return value.lower()

class DocumentConfig(BaseModel):
    CHUNK_SIZE: int = 768
    CHUNK_OVERLAP: int = 70
    FILE_TYPES: List[str] = ['pdf', 'txt', 'csv', 'md', 'docx']
    PARSE_TABLES: bool = True
    GENERATE_SUMMARY: bool = False

class PathConfig(BaseModel):
    DOCUMENT_DIR: str = "./data"
    DOMAIN_CENTROID_PATH: str = "./domain_centroid.npy"

class EnvironmentConfig(BaseModel):
    DOMAIN_KEYWORDS: List[str] = []
    AUTO_DOMAIN_KEYWORDS: List[str] = []
    USER_ADDED_KEYWORDS: List[str] = []

    @property
    def merged_keywords(self) -> List[str]:
        domain = self.DOMAIN_KEYWORDS or []
        auto = self.AUTO_DOMAIN_KEYWORDS or []
        user = self.USER_ADDED_KEYWORDS or []
        return sorted(list(set(domain + auto + user))) # Sort for consistency

# === Root Config Model ===
class AppConfig(BaseModel):
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    # Automatically update derived env fields after validation/loading
    #@model_validator(mode='after')
    #def update_derived_env_fields(self) -> 'AppConfig':
     #   self.env.API_TIMEOUT = self.security.API_TIMEOUT
     #   self.env.EMBEDDING_MODEL = self.model.EMBEDDING_MODEL
     #   return self

    @classmethod
    def load_from_yaml(cls, path: Path = CONFIG_YAML_PATH) -> 'AppConfig':
        """Loads configuration from YAML, validates, handles defaults, and returns instance."""
        if not path.exists():
            logger.warning(f"Configuration file {path} not found. Creating default config.")
            default_config = cls()
            try:
                # Use model_dump with exclude_defaults=False? Or just dump? Exclude keys explicitly.
                dump_data = default_config.model_dump(exclude={'security': {'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'}})
                save_yaml_config(dump_data, path)
            except Exception as save_e:
                logger.error(f"Failed to save default config file {path}: {save_e}")
            return default_config

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_config_data = yaml.safe_load(f) or {}

            # Validate raw data against the AppConfig model
            # Pydantic automatically merges defaults for missing fields
            instance = cls(**raw_config_data)

            # Note: The @model_validator handles updating derived fields now.

            logger.info(f"Configuration loaded and validated from {path}")
            return instance
        except Exception as e: # Catch YAML, Validation, and other errors
            logger.error(f"Failed to load/validate configuration from {path}: {e}", exc_info=True)
            raise RuntimeError(f"Could not load/validate configuration from {path}") from e

    def update_and_save(self, updates: Dict[str, Any], path: Path = CONFIG_YAML_PATH):
        """Updates current config, validates, saves if changed."""
        logger.info("Received config updates. Validating and applying...")
        config_changed = False

        # Get current state as dict for comparison later
        # Use model_dump() for Pydantic v2
        dump_method = getattr(self, 'model_dump', getattr(self, 'dict', None))
        if not dump_method: raise AttributeError("Cannot find dump method")
        original_dump = dump_method()

        # Create a dictionary representing the potential new state by deep merging
        # This avoids potential issues with model_copy and ensures we work with dicts first
        potential_new_state_dict = original_dump.copy()
        for section_key, section_updates in updates.items():
            if section_key in potential_new_state_dict and isinstance(section_updates, dict):
                 # Merge the update into the existing section dictionary
                potential_new_state_dict[section_key] = {
                     **potential_new_state_dict.get(section_key, {}),
                     **section_updates
                 }
            elif section_key in potential_new_state_dict:
                 # Handle cases where update is not a dict? Or log warning?
                 logger.warning(f"Update for section '{section_key}' is not a dict, skipping merge.")
            # else: section_key not in original, ignore?

        try:
            # --- *** VALIDATE THE FULL POTENTIAL DICT *** ---
            validated_new_config = AppConfig(**potential_new_state_dict)
            # --- *** END VALIDATION *** ---

            # Convert validated config back to dict for comparison and saving
            validated_new_dump = validated_new_config.model_dump()

            # Compare the validated new dictionary state with the original one
            if original_dump != validated_new_dump:
                config_changed = True
                logger.info("Configuration data changed.")

                # --- *** UPDATE SELF (CFG) FROM VALIDATED OBJECT *** ---
                # Assign the validated nested objects back to self
                for field_name in self.model_fields.keys():
                     if hasattr(validated_new_config, field_name):
                         setattr(self, field_name, getattr(validated_new_config, field_name))
                # --- *** END UPDATE SELF *** ---

            else:
                logger.info("No effective configuration changes detected.")

        except ValidationError as e:
            logger.error(f"Validation error applying updates: {e}")
            raise # Re-raise validation error

        if config_changed:
            logger.info("Configuration changed. Attempting save...")
            try:
                # Dump the *validated* new state, excluding API keys
                config_dict_to_save = validated_new_config.model_dump(exclude={'security': {'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'}})
                save_yaml_config(config_dict_to_save, path) # Call utility save function
                logger.info("Configuration save successful.")
            except Exception as e:
                logger.error(f"Error during configuration save call: {e}", exc_info=True)
                raise # Re-raise save error
        else:
            logger.info("Skipping save as no changes were detected.")

        return config_changed

# --- Initialize Singleton Configuration Instance ---
# ... (keep cfg initialization as before) ...
try:
    cfg = AppConfig.load_from_yaml()
    # ... (log key status) ...
except Exception:
    logging.critical("CRITICAL: Failed to load configuration...", exc_info=True)
    cfg = None # Set cfg to None if loading fails critically

# --- End config.py ---
