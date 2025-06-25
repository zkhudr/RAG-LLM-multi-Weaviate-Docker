# config.py (Pydantic v2 strict schema, legacy aliases, and provider-default fixes)

import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Path to configuration YAML
CONFIG_YAML_PATH = Path("./config_settings.yaml").resolve()

# --- YAML Flow Style for Lists ---
class FlowStyleList(list):
    pass

def flow_style_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class ConfigDumper(yaml.SafeDumper):
    pass

ConfigDumper.add_representer(FlowStyleList, flow_style_list_representer)
# --- End Flow Style ---


def save_yaml_config(data: Dict[str, Any], path: Path) -> bool:
    """Atomically save validated config data dictionary to YAML file."""
    temp_path = path.with_suffix(".yaml.tmp")
    try:
        dump_data = data.copy()
        if 'env' in dump_data:
            env_section = dump_data['env']
            for key in ['DOMAIN_KEYWORDS', 'AUTO_DOMAIN_KEYWORDS', 'USER_ADDED_KEYWORDS']:
                if key in env_section and isinstance(env_section[key], list):
                    env_section[key] = FlowStyleList(env_section[key])

        # Write to temp file
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                dump_data,
                f,
                default_flow_style=None,
                sort_keys=False,
                Dumper=ConfigDumper,
                indent=2
            )
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace
        try:
            os.replace(temp_path, path)
        except OSError as e:
            logger.warning(f"Atomic replace failed: {e}; writing directly.")
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    dump_data,
                    f,
                    default_flow_style=None,
                    sort_keys=False,
                    Dumper=ConfigDumper,
                    indent=2
                )
        finally:
            # Always try to remove the temp file; ignore if it's gone
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

        logger.info(f"Configuration saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save config: {e}", exc_info=True)
        # Cleanup temp file on error
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e2:
            logger.warning(f"Could not delete temp file after error {temp_path}: {e2}")
        raise


# === Configuration Models ===

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_history_turns: int = Field(5)


class DomainKeywordExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keybert_model: str = Field('all-MiniLM-L6-v2')
    top_n_per_doc: int = Field(10)
    final_top_n: int = Field(100)
    min_doc_freq_mode: Literal['absolute', 'fraction'] = Field('absolute')
    min_doc_freq_abs: Optional[int] = Field(None)
    min_doc_freq_frac: Optional[float] = Field(None)
    diversity: float = Field(0.7)
    no_pos_filter: bool = Field(False)
    # Legacy aliases
    min_doc_freq: Optional[int] = Field(None, alias='min_doc_freq')
    extraction_diversity: Optional[float] = Field(None, alias='extraction_diversity')

    @model_validator(mode='after')
    def apply_aliases(self):
        if self.min_doc_freq is not None:
            if self.min_doc_freq_mode == 'absolute':
                object.__setattr__(self, 'min_doc_freq_abs', self.min_doc_freq)
            else:
                object.__setattr__(self, 'min_doc_freq_frac', float(self.min_doc_freq))
        if self.extraction_diversity is not None:
            object.__setattr__(self, 'diversity', self.extraction_diversity)
        return self


class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    SANITIZE_INPUT: bool = True
    RATE_LIMIT: int = 10
    API_TIMEOUT: int = 40
    CACHE_ENABLED: bool = False

    DEEPSEEK_API_KEY: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""), exclude=True)
    OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""), exclude=True)
    ANTHROPIC_API_KEY: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""), exclude=True)
    COHERE_API_KEY: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""), exclude=True)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    COLLECTION_NAME: str = "Industrial_tech"
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
    PERFORM_TECHNICAL_VALIDATION: bool = True
    WEAVIATE_HOST: str = os.getenv("WEAVIATE_DOCKER_HOST", "localhost")
    WEAVIATE_HTTP_PORT: int = int(os.getenv("WEAVIATE_HOST_HTTP_PORT", 8080))
    WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_HOST_GRPC_PORT", 50051))
    retrieve_with_history: bool = False
    WEAVIATE_TIMEOUT: Tuple[int, int] = (10, 120)

    @field_validator('SEARCH_TYPE')
    def validate_search_type_cls(cls, value):
        allowed = {'mmr', 'similarity', 'similarity_score_threshold', 'hybrid'}
        low = value.lower()
        if low not in allowed:
            raise ValueError(f"SEARCH_TYPE must be one of {allowed}, got '{value}'")
        return low


ApiProvider = Literal["deepseek", "openai", "anthropic", "cohere", "none"]

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # External API endpoints
    PROVIDER_URLS: Dict[str, str] = Field(
        default_factory=lambda: {
            "deepseek":  "https://api.deepseek.com/v1/chat/completions",
            "openai":    "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/complete",
            "cohere":    "https://api.cohere.com/v1/completions"
        }
    )
    # Default model names per provider
    EXTERNAL_API_MODEL_DEFAULTS: Dict[str, str] = Field(
        default_factory=lambda: {
            "openai":    "gpt-3.5-turbo",
            "anthropic": "claude-v1",
            "cohere":    "command-nightly",
            # Use the chat model name that invokes DeepSeek-V3
            "deepseek":  "deepseek-chat"
        }
    )

    LLM_TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 1536
    OLLAMA_MODEL: str = "deepseek-coder:6.7b-instruct-q5_K_M"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    TOP_P: float = 0.9
    FREQUENCY_PENALTY: float = 0.1
    SYSTEM_MESSAGE: str = Field("You are an expert AI assistant...")
    EXTERNAL_API_PROVIDER: ApiProvider = "deepseek"
    EXTERNAL_API_MODEL_NAME: Optional[str] = None
    MERGE_STRATEGY: Literal["api_first", "concat", "local_only"] = "api_first"

    @field_validator('EXTERNAL_API_PROVIDER')
    def validate_provider(cls, value):
        return value.lower()

    @model_validator(mode='after')
    def check_api_keys(self):
        if self.EXTERNAL_API_PROVIDER != 'none':
            mapping = {
                'deepseek': 'DEEPSEEK_API_KEY',
                'openai':   'OPENAI_API_KEY',
                'anthropic':'ANTHROPIC_API_KEY',
                'cohere':   'COHERE_API_KEY'
            }
            key = mapping.get(self.EXTERNAL_API_PROVIDER)
            if key and not os.getenv(key):
                object.__setattr__(self, 'EXTERNAL_API_PROVIDER', 'none')
        return self

class DocumentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    CHUNK_SIZE: int = Field(768)
    CHUNK_OVERLAP: int = Field(70)
    MIN_CONTENT_LENGTH: int = Field(50)
    FILE_TYPES: List[str] = Field(['pdf', 'txt', 'csv', 'md', 'docx'])
    PARSE_TABLES: bool = Field(True)
    GENERATE_SUMMARY: bool = Field(False)

class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    DOCUMENT_DIR: str = "./data"
    DOMAIN_CENTROID_PATH: str = "./domain_centroid.npy"
    CENTROID_DIR: str = "./centroids"

class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    DOMAIN_KEYWORDS: List[str] = Field(default_factory=list)
    AUTO_DOMAIN_KEYWORDS: List[str] = Field(default_factory=list)
    USER_ADDED_KEYWORDS: List[str] = Field(default_factory=list)
    SELECTED_N_TOP: int = Field(10000)

    @property
    def merged_keywords(self) -> List[str]:
        return sorted(set(self.DOMAIN_KEYWORDS + self.AUTO_DOMAIN_KEYWORDS + self.USER_ADDED_KEYWORDS))

class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    CENTROID_AUTO_THRESHOLD: float = Field(0.5)
    CENTROID_DIVERSITY_THRESHOLD: float = Field(0.01)
    CENTROID_UPDATE_MODE: Literal['auto', 'manual'] = Field('auto')
    MIN_QUALITY_SCORE: float = Field(0.3)
    # Legacy alias
    CENTROID_DIVERSITY: Optional[float] = Field(None, alias='CENTROID_DIVERSITY')

    @model_validator(mode='after')
    def apply_legacy_alias(self):
        if self.CENTROID_DIVERSITY is not None:
            object.__setattr__(self, 'CENTROID_DIVERSITY_THRESHOLD', self.CENTROID_DIVERSITY)
        return self

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    security: SecurityConfig                   = Field(default_factory=SecurityConfig)
    retrieval: RetrievalConfig                 = Field(default_factory=RetrievalConfig)
    model: ModelConfig                         = Field(default_factory=ModelConfig)
    document: DocumentConfig                   = Field(default_factory=DocumentConfig)
    paths: PathConfig                          = Field(default_factory=PathConfig)
    env: EnvironmentConfig                     = Field(default_factory=EnvironmentConfig)
    pipeline: PipelineConfig                   = Field(default_factory=PipelineConfig)
    ingestion: IngestionConfig                 = Field(default_factory=IngestionConfig)
    domain_keyword_extraction: DomainKeywordExtractionConfig = Field(default_factory=DomainKeywordExtractionConfig)

    @classmethod
    def load_from_yaml(cls, path: Path = CONFIG_YAML_PATH) -> "AppConfig":
        if not path.exists():
            default = cls()
            data = default.model_dump(exclude={'security': {
                'DEEPSEEK_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY','COHERE_API_KEY'
            }})
            save_yaml_config(data, path)
            return default

        try:
            raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as ye:
            logger.error(f"YAML parsing error: {ye}", exc_info=True)
            raise RuntimeError("Could not parse YAML config") from ye

        inst = cls(**raw)
        logger.info(f"Configuration loaded from {path}")
        return inst

    def update_and_save(self, updates: Dict[str, Any], path: Path = CONFIG_YAML_PATH) -> bool:
        original = self.model_dump()
        merged = {**original, **updates}
        validated = AppConfig(**merged)
        new_dump = validated.model_dump()
        if new_dump != original:
            save_yaml_config(new_dump, path)
            for field in type(self).model_fields:
                setattr(self, field, getattr(validated, field))
            return True
        return False

    def reload(self):
            """Reloads the configuration from the YAML file path."""
            with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
                new_data = yaml.safe_load(f) or {}
            self.__init__(**new_data)

# Instantiate config
try:
    cfg = AppConfig.load_from_yaml()
except Exception:
    logger.critical("CRITICAL: Failed to load configuration...", exc_info=True)
    cfg = None
