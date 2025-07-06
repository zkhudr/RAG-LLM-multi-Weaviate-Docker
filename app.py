# Disable optional telemetry as early as possible
import os
import io
os.environ['POSTHOG_DISABLED'] = 'true'
import sys
# ─── Standard library ────────────────────────────────────────────────────────────────
import csv
import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import yaml
import pipeline
import numpy as np
from datetime import datetime
from pathlib import Path
from threading import Thread, Timer
from typing import Any, Dict, List, Optional

# ─── Third-party libraries ──────────────────────────────────────────────────────────
import docker
import matplotlib
matplotlib.use('Agg')  # must be set before pyplot
import matplotlib.pyplot as plt
import requests
import weaviate
from dotenv import load_dotenv
from flask import (
    current_app, Flask, g, jsonify, render_template,
    request, send_file, session, url_for
)
from flask_login import LoginManager
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, Field, ValidationError
from requests.exceptions import RequestException
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from weaviate.exceptions import WeaviateConnectionError


# ─── Local application modules ─────────────────────────────────────────────────────
from calculate_centroid import calculate_and_save_centroid
from centroid_manager import (
    CentroidManager, centroid_exists, get_centroid_stats,
    should_recalculate_centroid
)
from config import cfg, CONFIG_YAML_PATH, save_yaml_config, AppConfig
if cfg is None:
    logger.critical("Config failed to load; exiting.")
    sys.exit(1)

from ingest_block import run_ingestion as run_incremental_ingestion
from pipeline import (
    initialize_pipeline,
    init_pipeline_once,
    is_pipeline_valid,
    PIPELINE_INSTANCE
)
from datetime import datetime
from validate_configs import main as validate_configs


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 💀 Kill previous handlers
if logger.hasHandlers():
    logger.handlers.clear()




# ─── Validate config files at startup ────────────────────────────────────────────────
try:
    validate_configs()
except SystemExit as e:
    if e.code != 0:
        logging.critical(
            f"Configuration validation failed (exit code {e.code}). Shutting down."
        )
        sys.exit(e.code)

# ─── Load environment variables ───────────────────────────────────────────────────────
load_dotenv()

# ─── API key status cache ────────────────────────────────────────────────────────────
API_KEY_STATUS_CACHE = {
    "deepseek":  bool(os.getenv("DEEPSEEK_API_KEY")),
    "openai":    bool(os.getenv("OPENAI_API_KEY")),
    "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    "cohere":    bool(os.getenv("COHERE_API_KEY")),
}

# ─── Flask application setup ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "fallback_secret_key_for_dev_only"
)
if app.secret_key == "fallback_secret_key_for_dev_only":
    print(
        "WARNING: Using fallback Flask secret key. "
        "Set FLASK_SECRET_KEY environment variable for production."
    )

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return None  # No login system implemented yet

# Session store
app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_FILE_DIR="./.flask_session",
    SESSION_FILE_THRESHOLD=100,
)
Session(app)

# Database (SQLite chat history)
db = SQLAlchemy()
db_path = Path(app.instance_path) / "chat_history.db"
app.config.update(
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{db_path}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

# ─── Logging setup ──────────────────────────────────────────────────────────────────
log_level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Ensure instance folder exists early
try:
    os.makedirs(app.instance_path, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Could not create instance path '{app.instance_path}': {e}", file=sys.stderr)

log_file = Path(app.instance_path) / "app.log"

file_handler = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(formatter)
file_handler.setLevel(log_level)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(log_level)

# Root logger
root = logging.getLogger()
root.setLevel(log_level)
root.handlers.clear()
root.addHandler(file_handler)
root.addHandler(console_handler)

# Suppress noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("weaviate").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Logging configured: Level=%s, File=%s", logging.getLevelName(log_level), log_file)

# ─── Docker client for multi-instance control ───────────────────────────────────────
docker_available = False
docker_client = None
try:
    docker_client = docker.from_env()
    docker_client.ping()
    docker_available = True
    logger.info("Docker client initialized and connected successfully.")
except Exception as err:
    logger.error(f"Failed to initialize Docker client: {err}. Multi-instance disabled.")

# ─── Shared constants ────────────────────────────────────────────────────────────────
WEAVIATE_STATE_FILE     = Path("./weaviate_state.json").resolve()
DEFAULT_BASE_HTTP_PORT  = 8090
DEFAULT_BASE_GRPC_PORT  = 51001
WEAVIATE_IMAGE          = "semitechnologies/weaviate:1.25.1"
WEAVIATE_DATA_DIR_HOST  = Path("./weaviate_data").resolve()
SAVE_LOCK               = threading.Lock()



class KeywordBuilderRequest(BaseModel):
    keybert_model: str = Field(
        default_factory=lambda: cfg.domain_keyword_extraction.keybert_model,
        alias='keybert_model'
    )
    top_n_per_doc: int = Field(
        default_factory=lambda: cfg.domain_keyword_extraction.top_n_per_doc,
        alias='top_n_per_doc'
    )
    final_top_n: int = Field(
        default_factory=lambda: cfg.domain_keyword_extraction.final_top_n,
        alias='final_top_n'
    )
    min_doc_freq: int = Field(
        default_factory=lambda: getattr(cfg.domain_keyword_extraction, 'min_doc_freq_abs', 2),
        alias='min_doc_freq_abs'
    )
    extraction_diversity: float = Field(
        default_factory=lambda: cfg.domain_keyword_extraction.diversity,
        alias='extraction_diversity'
    )
    no_pos_filter: bool = Field(
        default=False,
        alias='no_pos_filter'
    )
    timeout: int = Field(
        default_factory=lambda: getattr(cfg.security, 'APITIMEOUT', 50000)
    )

    class Config:
         allow_population_by_field_name = True
         extra = 'ignore'    # silently drop any unknown fields

# --- Database Model ---
class SavedChat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    history = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- Constants ---
CONFIG_YAML_PATH = Path("./config_settings.yaml").resolve()
PRESETS_FILE = Path("./presets.json").resolve()

# --- Import Pydantic ValidationError ---
try:
    from pydantic import ValidationError
except ImportError:
    try: from pydantic.error_wrappers import ValidationError
    except ImportError: logger.error("Pydantic ValidationError could not be imported."); ValidationError = Exception

 # --- Import Project Modules (Config, Pipeline) ---
try:
     from config import cfg, CONFIG_YAML_PATH, save_yaml_config, AppConfig
     logger.info("Successfully imported configuration (cfg, AppConfig).")
except ImportError as e:
     logger.critical(f"CRITICAL: Failed to import configuration modules: {e}", exc_info=True)
     sys.exit(1)

if cfg is None:
    logger.critical("CRITICAL: Configuration loaded but 'cfg' is None. Diagnosing issue…")
     # 1) Can we instantiate defaults?
    try:
         _ = AppConfig()
         logger.critical("Default AppConfig instantiated successfully; loading from YAML failed.")
    except Exception as default_error:
         logger.critical(f"Failed to create default AppConfig: {default_error}", exc_info=True)

     # 2) Inspect raw YAML for missing top-level sections
    import yaml
    try:
         with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f:
             raw = yaml.safe_load(f) or {}
         missing = [sec for sec in AppConfig.model_fields if sec not in raw]
         for sec in missing:
             logger.critical(f"Missing section in YAML: '{sec}'")
    except Exception as yaml_error:
         logger.critical(f"Failed to inspect raw YAML: {yaml_error}", exc_info=True)

    logger.critical("Configuration diagnosis complete. Exiting application.")
    sys.exit(1)

# ---  Ingestion Script Imports ---
ingest_full_available = False
ingest_block_available = False
DocumentProcessor = None
PipelineConfig = None       # Use this name (from ingest_docs_v7)
run_incremental_ingestion = None # Use this name (from ingest_block)

try:
    # Import DocumentProcessor, PipelineConfig, setup_logging from v7
    from ingest_docs_v7 import DocumentProcessor, PipelineConfig 
    ingest_full_available = True
    logger.info("Successfully imported full ingestion components (DocumentProcessor, PipelineConfig).")
except ImportError as e:
    logger.error(f"Failed to import full ingestion components from ingest_docs_v7.py: {e}. Full ingestion disabled.", exc_info=False)
    # Set all related variables to None on failure
    DocumentProcessor, PipelineConfig, setup_logging = None, None

try:
    # Import run_ingestion function from ingest_block (alias it for clarity in routes)
    from ingest_block import run_ingestion  # the top-level function
    run_incremental_ingestion = run_ingestion # Assign to the expected variable name
    ingest_block_available = True
    logger.info("Successfully imported incremental ingestion component (run_ingestion as run_incremental_ingestion).")
except ImportError as e:
    logger.error(f"Failed to import 'run_ingestion' from ingest_block.py: {e}. Incremental ingestion disabled.", exc_info=False)
    run_incremental_ingestion = None # Ensure it's None if import fails
# --- End Corrected Ingestion Imports ---

# --- Ensure Instance/Session/Data Dirs ---
try:
    app.instance_path_obj = Path(app.instance_path)
    app.instance_path_obj.mkdir(parents=True, exist_ok=True)
    Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
    WEAVIATE_DATA_DIR_HOST.mkdir(parents=True, exist_ok=True)
    if cfg and cfg.paths and cfg.paths.DOCUMENT_DIR: # Ensure doc dir from config if available
         Path(cfg.paths.DOCUMENT_DIR).resolve().mkdir(parents=True, exist_ok=True)
    logger.info("Checked/Ensured instance, session, Weaviate data, and document directories.")
except Exception as e: logger.error(f"CRITICAL: Could not create required directories: {e}"); exit(1)

# --- Global Presets Variable ---
presets = {}


# === Utility: Reload config and re-init pipeline ===

def _wait_for_weaviate_rest(host, port, retries=10, delay=0.2):
    import time
    import httpx
    url = f"http://{host}:{port}/v1/meta"
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                app.logger.info(f"Weaviate REST is ready (attempt {attempt})")
                return True
        except Exception:
            app.logger.debug(f"Attempt {attempt}: Weaviate not ready yet.")
        time.sleep(delay)
    app.logger.error(f"Weaviate REST failed to start after {retries} attempts.")
    return False


def reload_and_maybe_reinit(ctx, override_weaviate: bool = False):
    with ctx:
        try:
            # 1) Reload YAML-backed cfg
            cfg.reload()

            # Only override if explicitly requested
            if override_weaviate:
                cfg.retrieval.WEAVIATE_HOST = "localhost"
                cfg.retrieval.WEAVIATE_HTTP_PORT = 8092
                cfg.retrieval.WEAVIATE_GRPC_PORT = 51003
                cfg.retrieval.WEAVIATE_ALIAS = "Main"
                logger.info("Weaviate override applied after config reload.")

            logger.info("Reloaded config from disk.")

            # 🔹 Safely close any existing pipeline
            global PIPELINE_INSTANCE
            if PIPELINE_INSTANCE:
                try:
                    PIPELINE_INSTANCE.close()
                    logger.info("Previous pipeline closed cleanly.")
                except Exception as e:
                    logger.warning(f"Error closing previous pipeline: {e}")
                PIPELINE_INSTANCE = None  # 🟢 IMPORTANT: clear reference

            # 2) Always re-init pipeline (force new instance)
            init_pipeline_once(force=True)

        except Exception as reload_err:
            logger.error("Reload or pipeline init failed.", exc_info=True)

@app.route("/delete_collection", methods=["POST"])
def delete_collection():
    """Delete a collection from Weaviate."""
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Collection name required."}), 400

    try:
        import weaviate

        client = weaviate.connect_to_custom(
            http_host=cfg.retrieval.WEAVIATE_HOST,
            http_port=cfg.retrieval.WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=cfg.retrieval.WEAVIATE_HOST,
            grpc_port=cfg.retrieval.WEAVIATE_GRPC_PORT,
            grpc_secure=False,
            skip_init_checks=True
        )

        client.collections.delete(name)
        app.logger.info(f"Deleted collection '{name}' successfully.")
        return jsonify({"success": True})
    except Exception as e:
        app.logger.error(f"Error deleting collection '{name}'", exc_info=True)
        return jsonify({"error": str(e)})

@app.route("/create_collection", methods=["POST"])
def create_collection():
    """
    Creates a new empty collection in the active Weaviate instance.
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Collection name required."}), 400

    try:
        import weaviate

        # Always create client using current cfg to respect active instance
        client = weaviate.connect_to_custom(
            http_host=cfg.retrieval.WEAVIATE_HOST,
            http_port=cfg.retrieval.WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=cfg.retrieval.WEAVIATE_HOST,
            grpc_port=cfg.retrieval.WEAVIATE_GRPC_PORT,
            grpc_secure=False,
            skip_init_checks=True
        )

        existing_names = client.collections.list_all()
        if name in existing_names:
            return jsonify({"error": "Collection already exists."}), 400

        client.collections.create(name)
        logger.info(f"Created collection '{name}' successfully.")
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Error creating collection '{name}': {e}", exc_info=True)
        return jsonify({"error": str(e)})






# === Utility Functions ===


def safe_int(value, default=0, min_val=None, max_val=None) -> int:
    """Safely convert to int with bounds checking"""
    try:
        val_str = str(value).strip()
        if not val_str: num = default
        else: num = int(val_str)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert '{value}' to int, using default {default}")
        return default

    if min_val is not None and num < min_val:
        logging.warning(f"Int value {num} below min {min_val}, clamping to {min_val}")
        return min_val
    if max_val is not None and num > max_val:
        logging.warning(f"Int value {num} above max {max_val}, clamping to {max_val}")
        return max_val
    return num

def safe_float(value, default=0.0, min_val=None, max_val=None) -> float:
    """Safely convert to float with bounds checking"""
    try:
        val_str = str(value).strip()
        if not val_str: num = default
        else: num = float(val_str)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert '{value}' to float, using default {default}")
        return default

    if min_val is not None and num < min_val:
        logging.warning(f"Float value {num} below min {min_val}, clamping to {min_val}")
        return min_val
    if max_val is not None and num > max_val:
        logging.warning(f"Float value {num} above max {max_val}, clamping to {max_val}")
        return max_val
    return num

def safe_split(value, delimiter=',') -> List[str]:
    """Safely split and clean string inputs into a list."""
    if not isinstance(value, str): return []
    return [item.strip() for item in value.split(delimiter) if item.strip()]

def load_presets(filename=PRESETS_FILE):
    """Loads presets from JSON file and updates global variable."""
    global presets
    if not filename.exists():
        logger.warning(f"Presets file '{filename}' not found. Using empty presets.")
        presets = {}
        return presets
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            if not isinstance(loaded_data, dict): # Add type check
                raise TypeError("Presets file does not contain a valid JSON object (dictionary).")
            presets = loaded_data
            logger.info(f"Loaded {len(presets)} presets from '{filename}'. Global updated.")
            return presets
    except (json.JSONDecodeError, IOError, TypeError) as e: # Catch more errors
        logger.error(f"CRITICAL: Error loading presets from '{filename}': {e}. Returning empty presets.", exc_info=True)
        presets = {} # Reset global on error
        return presets
    

@app.route("/save_preset", methods=["POST"])
def save_preset_api():
    global presets

    # Ensure configuration is loaded
    if cfg is None:
        return jsonify({"success": False, "error": "Configuration unavailable."}), 503

    try:
        data = request.get_json() or {}
        preset_name = data.get("preset_name", "").strip()
        if not preset_name:
            return jsonify({"success": False, "error": "Preset name is required."}), 400

        preset_data = data.get("config")
        if not isinstance(preset_data, dict):
            return jsonify({"success": False, "error": "Invalid or missing preset data."}), 400

        # Remove illegal top-level 'presets' key if present
        preset_data.pop("presets", None)

        # Validate against AppConfig model
        validated = AppConfig(**preset_data)

        # 🚨 Sanitize before saving
        sanitized_dump = validated.model_dump_sanitized()

        # Save in-memory and to disk
        presets[preset_name] = sanitized_dump
        if not save_presets(presets):
            return jsonify({"success": False, "error": "Failed to write presets file."}), 500

        return jsonify({"success": True, "message": f"Preset '{preset_name}' saved."}), 200

    except ValidationError as ve:
        logger.error(f"Preset data validation failed: {ve}")
        return jsonify({"success": False, "error": "Invalid preset data."}), 400

    except Exception as e:
        logger.error(f"Preset save API error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# === NEW: Multi-Instance Helper Functions ===

def save_config_to_yaml(config_dict: Dict[str, Any]) -> bool:
    """Validates and saves the config dictionary to the YAML file."""
    global cfg  # Ensure we update the global cfg instance

    # Guard against missing config
    if cfg is None:
        logging.error("Cannot save YAML: 'cfg' is None.")
        raise RuntimeError("Configuration system not loaded.")

    try:
        # 1. Validate the incoming dictionary by creating a new AppConfig instance
        validated_config = AppConfig(**config_dict)

        # 2. Replace the global 'cfg' with the validated config model
        cfg = validated_config

        # 3. Prepare the dictionary for YAML dump
        dump_dict = validated_config.model_dump_sanitized() # Pydantic v2

        # 4. Atomic write to YAML
        temp_path = f"{CONFIG_YAML_PATH}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(dump_dict, f, indent=2, sort_keys=False)
        os.replace(temp_path, CONFIG_YAML_PATH)

        logging.info(f"Configuration saved successfully to {CONFIG_YAML_PATH}")
        return True

    except ValidationError as e:
        logging.error(f"Configuration validation failed: {e}")
        raise

    except Exception as e:
        logging.error(f"Failed to save YAML configuration: {e}", exc_info=True)
        raise RuntimeError(f"Config save failed: {str(e)}")


def get_new_auto_keywords(filename="new_auto_keywords.txt") -> List[str]:
        """Reads keywords from the new_auto_keywords.txt file."""
        try:
            filepath = Path(filename).resolve()
            if not filepath.exists():
                logging.warning(f"Auto keywords file '{filepath}' not found.")
                return []
            with open(filepath, "r", encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return [kw.strip() for kw in content.split(",") if kw.strip()]
            return []
        except Exception as e:
            logging.error(f"Error reading auto keywords file '{filename}': {e}")
            return []
        
def load_weaviate_state() -> Dict:
        """Loads the state of managed Weaviate instances from JSON file."""
        if WEAVIATE_STATE_FILE.exists():
            try:
                with open(WEAVIATE_STATE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e: # pragma: no cover
                app.logger.error(f"Error loading Weaviate state from {WEAVIATE_STATE_FILE}: {e}")
                return {} # Return empty dict on error
        return {} # Return empty dict if file doesn't exist

def save_weaviate_state(state: Dict):
        """Saves the state of managed Weaviate instances to JSON file."""
        try:
            # Ensure parent directory exists
            WEAVIATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(WEAVIATE_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            app.logger.info(f"Saved Weaviate state to {WEAVIATE_STATE_FILE}")
        except IOError as e: # pragma: no cover
            app.logger.error(f"Error saving Weaviate state to {WEAVIATE_STATE_FILE}: {e}")

def get_next_ports(existing_instances: Dict) -> tuple[int, int]:
        """Calculates the next available HTTP and gRPC ports."""
        max_http = DEFAULT_BASE_HTTP_PORT - 1
        max_grpc = DEFAULT_BASE_GRPC_PORT - 1
        if not existing_instances: # Handle empty state
            return DEFAULT_BASE_HTTP_PORT, DEFAULT_BASE_GRPC_PORT

        for data in existing_instances.values():
            # Safely get ports, fallback to default - 1 if missing/invalid
            try: http_port = int(data.get('http_port', DEFAULT_BASE_HTTP_PORT - 1))
            except (ValueError, TypeError): http_port = DEFAULT_BASE_HTTP_PORT - 1
            try: grpc_port = int(data.get('grpc_port', DEFAULT_BASE_GRPC_PORT - 1))
            except (ValueError, TypeError): grpc_port = DEFAULT_BASE_GRPC_PORT - 1

            max_http = max(max_http, http_port)
            max_grpc = max(max_grpc, grpc_port)
        return max_http + 1, max_grpc + 1

def get_active_instance_name() -> Optional[str]:
        """Finds the instance name matching the current cfg connection details."""
        if cfg is None: return None
        current_host = cfg.retrieval.WEAVIATE_HOST
        current_http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        #logger = app.logger # Use app logger

        # 1. Check managed instances first
        try:
            instance_state = load_weaviate_state()
            for name, details in instance_state.items():
                # Ensure ports are compared as integers
                try:
                    details_http_port = int(details.get('http_port', -1))
                except (ValueError, TypeError):
                    continue # Skip if port is invalid
                if details.get('host') == current_host and details_http_port == current_http_port:
                    logger.debug(f"Active instance matches managed instance: '{name}'")
                    return name
        except Exception as e: # pragma: no cover
            logger.error(f"Error reading Weaviate state file for active check: {e}")

        # 2. If no match in state, check if it matches the default FROM THE YAML FILE
        try:
            with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            default_host = yaml_config.get('retrieval', {}).get('WEAVIATE_HOST', 'localhost')
            # Safely convert default ports from file to int
            try: default_http_port = int(yaml_config.get('retrieval', {}).get('WEAVIATE_HTTP_PORT', 8080))
            except (ValueError, TypeError): default_http_port = 8080

            if current_host == default_host and current_http_port == default_http_port:
                logger.debug("Active instance matches default from config file.")
                return "Default (from config)"
        except Exception as e: # pragma: no cover
            logger.error(f"Could not read or parse default config file ({CONFIG_YAML_PATH}) for active check: {e}")

        # 3. If still no match, return None (or a placeholder if preferred)
        logger.warning(f"Could not determine active instance name for {current_host}:{current_http_port}. Not found in state or default config.")
        return None

    # === END NEW Helper Functions ===
# === Flask Routes ===

# --- Main Route & API Endpoints ---


@app.errorhandler(Exception)
def handle_all_errors(e):
    # Let Flask’s built-in HTTPExceptions (404, 403, etc.) pass through unmodified
    if isinstance(e, HTTPException):
        return e

    # Log and return JSON for any other uncaught exceptions
    current_app.logger.error(f"Uncaught error: {e}", exc_info=True)
    resp = jsonify({ "error": str(e) })
    resp.status_code = 500
    return resp


@app.route("/", methods=["GET"])
def index(): return render_template("index.html")

@app.route('/get_config', methods=['GET'])
def get_config_api():
    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503
    try:
        config_dict = {}

        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))
        if not dump_method: 
            raise TypeError("Config object cannot be serialized.")
        
        config_dict = dump_method(exclude={'security': {...}}, exclude_none=True)

        # 🧼 Cleanup floaters
        dke = config_dict.get('domain_keyword_extraction', {})
        if dke.get('extraction_diversity') is None:
            dke.pop('extraction_diversity', None)

        if 'pipeline' not in config_dict:
            logger.warning("Pipeline section missing in config. Adding default values.")
            config_dict['pipeline'] = {'max_history_turns': 5}

        # ✅ SAFE LOGGING (avoids CP1252 crash on Windows console)
        try:
            preview = json.dumps(config_dict, ensure_ascii=True)[:1000] + "..."
            logger.info(f"Config to be returned: {preview}")
        except Exception as log_err:
            logger.warning(f"Could not safely log config_dict: {log_err}")

        return jsonify(config_dict)

    except Exception as e: 
        logger.error(f"API /get_config error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route('/delete_preset/<path:preset_name>', methods=['DELETE'])
def delete_preset_api(preset_name):
    """API Endpoint: Deletes a specified preset."""
    global presets
    #logger = app.logger
    try:
        decoded_preset_name = urllib.parse.unquote(preset_name)
    except Exception as decode_err:
        logger.error(f"API /delete_preset: Error decoding preset name '{preset_name}': {decode_err}")
        return jsonify({"success": False, "error": "Invalid preset name encoding."}), 400

    logger.info(f"API /delete_preset: Request received for preset '{decoded_preset_name}'.")

    # --- Reload presets from file for safety before checking ---
    # Note: This ensures we check against the file, but might hide sync issues elsewhere
    current_presets_from_file = load_presets()
    logger.info(f"Reloaded presets from file. Keys before delete check: {list(current_presets_from_file.keys())}")
    # --- End Reload ---

    # Check if preset exists in the reloaded data
    if decoded_preset_name not in current_presets_from_file:
        logger.warning(f"API /delete_preset: Preset '{decoded_preset_name}' not found in reloaded file data.")
        # Send JSON 404
        return jsonify({"success": False, "error": f"Preset '{decoded_preset_name}' not found."}), 404

    try:
        # Create a copy to modify
        presets_after_delete = current_presets_from_file.copy()
        # Delete from the copy
        del presets_after_delete[decoded_preset_name]
        logger.info(f"API /delete_preset: Preset '{decoded_preset_name}' removed from copy. Attempting save...")

        # Attempt to save the modified dictionary (copy)
        if save_presets(presets_after_delete): # save_presets updates global AND writes file
            logger.info(f"API /delete_preset: Presets file updated successfully. Global 'presets' updated.")
            # Send JSON 200
            return jsonify({"success": True, "message": f"Preset '{decoded_preset_name}' deleted successfully."}), 200
        else:
            # Save failed - Log error but maybe return 500 as file write failed
            logger.error(f"API /delete_preset: Critical - Failed to save presets file after preparing deletion of '{decoded_preset_name}'.")
            # Send JSON 500
            return jsonify({"success": False, "error": "Preset deletion failed during file save."}), 500

    except KeyError: # Should not happen due to check above, but good practice
         logger.error(f"API /delete_preset: KeyError for '{decoded_preset_name}' during deletion attempt (unexpected).")
         return jsonify({"success": False, "error": f"Preset '{decoded_preset_name}' could not be deleted (internal error)."}), 500
    except Exception as e:
        logger.error(f"API /delete_preset: Unexpected error deleting preset '{decoded_preset_name}': {e}", exc_info=True)
        # Send JSON 500
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500



@app.route('/api/key_status', methods=['GET'])
def api_key_status():
    return jsonify(API_KEY_STATUS_CACHE), 200


# --- Action Endpoints (JSON based) ---
@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    # 1) Get or rebuild the pipeline instance
    inst = init_pipeline_once()
    if inst is None or not is_pipeline_valid(inst):
        logger.critical("[run_pipeline] Pipeline unavailable or invalid.")
        return jsonify({
            "role": "assistant",
            "text": "Error: Pipeline unavailable.",
            "error": True
        }), 503

    try:
        # 2) Parse and validate incoming query
        data = request.get_json(silent=True) or {}
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({
                "role": "assistant",
                "text": "Error: Missing or empty 'query'.",
                "error": True
            }), 400

        # 3) Call the pipeline
        chat_history = session.get("chat_history", [])
        result_dict = inst.generate_response(query=user_query, chat_history=chat_history)

        # DEBUG: Show the raw response dict
        print("=== FINAL RESPONSE ===")
        print(repr(result_dict))

        logger.info(f"[Pipeline] Query successful. Result: {result_dict}")

        # 4) Extract text from the result
        raw_response = result_dict.get("response", "")
        if isinstance(raw_response, dict):
            response_text = raw_response.get("response", "").strip()
        else:
            response_text = str(raw_response).strip()

        # 5) Build assistant message
        assistant_message = {
            "role": "assistant",
            "text": response_text or "Sorry, I couldn't generate a valid response.",
            "sources": result_dict.get("sources") or result_dict.get("source"),
            "error": result_dict.get("error", False),
            "timestamp": datetime.now().isoformat()
        }

        # 6) Persist chat history and return
        max_history = getattr(cfg.pipeline, "max_history_turns", 50)
        chat_history.append(assistant_message)
        if len(chat_history) > max_history * 2:
            # max_turns = 5 means 10 messages (5 turns * 2 messages)
            chat_history = chat_history[-max_history * 2:]
        session["chat_history"] = chat_history
        return jsonify(assistant_message), 200

    except Exception as e:
        logger.error("Error in /run_pipeline", exc_info=True)
        return jsonify({
            "role": "assistant",
            "text": f"Internal error: {e}",
            "error": True
        }), 500

@app.route('/upload_files', methods=['POST'])
def upload_files():
    # Ensure configuration is loaded
    if cfg is None:
        return jsonify({"success": False, "error": "Configuration unavailable."}), 503

    # Create upload directory
    try:
        upload_dir = Path(cfg.paths.DOCUMENT_DIR).resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as dir_e:
        return jsonify({"success": False, "error": f"Could not prepare upload directory: {dir_e}"}), 500

    # Process incoming files
    try:
        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"success": False, "error": "No files provided."}), 400

        saved_files: List[str] = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                target = upload_dir / filename
                file.save(target)
                saved_files.append(filename)

        return jsonify({"success": True, "files": saved_files}), 200

    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {e}"}), 500


@app.route("/apply_preset/<preset_name>", methods=["POST"])
def apply_preset(preset_name):
    """
    Applies the selected preset config (without overriding active Weaviate instance),
    persists changes, and reloads the pipeline if necessary.
    """
    global cfg, presets

    if preset_name not in presets:
        return jsonify({"success": False, "error": "Preset not found"}), 404
    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503

    try:
        # ─── Clone safely ───
        from copy import deepcopy
        preset_data = deepcopy(presets[preset_name])

        # ─── Protect active Weaviate settings ───
        preset_data.setdefault("retrieval", {})
        active_retrieval = getattr(cfg, "retrieval", {})
        # Unconditionally overwrite these
        preset_data["retrieval"]["WEAVIATE_HOST"] = getattr(active_retrieval, "WEAVIATE_HOST")
        preset_data["retrieval"]["WEAVIATE_HTTP_PORT"] = getattr(active_retrieval, "WEAVIATE_HTTP_PORT")
        preset_data["retrieval"]["WEAVIATE_GRPC_PORT"] = getattr(active_retrieval, "WEAVIATE_GRPC_PORT")
        preset_data["retrieval"]["WEAVIATE_ALIAS"] = getattr(active_retrieval, "WEAVIATE_ALIAS")

        # ─── Apply and persist ───
        config_changed = cfg.update_and_save(preset_data)
        current_app.logger.info(f"Applied preset '{preset_name}', changes written: {config_changed}")

        # ─── Background reload ───
        if config_changed:
            Thread(
                target=reload_and_maybe_reinit,
                args=(current_app.app_context(),),
                kwargs={"override_weaviate": False},  # Weaviate already protected
                daemon=True
            ).start()

        # ─── Final config dump (sanitized) ───
        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))
        current_config_dict = (
            dump_method(exclude={'security': {
                'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'
            }}) if dump_method else {}
        )

        return jsonify({
            "success": True,
            "message": f"Preset '{preset_name}' applied.",
            "config": current_config_dict
        })

    except ValidationError as e:
        current_app.logger.error(f"Preset validation error '{preset_name}': {e}")
        return jsonify({"success": False, "error": "Invalid preset format."}), 400

    except Exception as e:
        current_app.logger.error(f"Preset apply error '{preset_name}': {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/save_config", methods=["POST"])
def save_config():
    """
    Receives a full config payload from the UI, validates it,
    persists it to config_settings.yaml, and triggers a background
    pipeline reload if necessary.
    """
    logger = current_app.logger

    # 1. Parse payload
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"/save_config: could not parse JSON: {e}", exc_info=True)
        return jsonify(success=False, error="Invalid JSON"), 400

    # 2. STRIP INVALID COLLECTION before diff-check
    
    bad_col = data.get("retrieval", {}).get("COLLECTION_NAME", "")
    if bad_col:
        if not isinstance(bad_col, str) or not bad_col.strip():
            fallback = cfg.retrieval.COLLECTION_NAME
            logger.warning(
                f"[save_config] Rejecting empty COLLECTION_NAME, resetting to '{fallback}'"
            )
            data.setdefault("retrieval", {})["COLLECTION_NAME"] = fallback


    # 3. Validate payload
    try:
        validated = AppConfig(**data)
    except Exception as e:
        logger.error(f"/save_config validation error: {e}", exc_info=True)
        return jsonify(success=False, error=str(e)), 400

    # 4. Check for diffs and persist
    try:
        before = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
        changed_keys = [
            key for key in data
            if before.get(key) != data.get(key)
        ]

        if not changed_keys:
            logger.info("/save_config: No actual changes in config. Skipping save.")
            return jsonify(
                success=True,
                changed=False,
                message="No config changes detected.",
                config=validated.model_dump()
            ), 200

        changed = cfg.update_and_save(data)
        cfg.reload()
        cfg.retrieval.WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
        cfg.retrieval.WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8092"))
        cfg.retrieval.WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "51003"))
        cfg.retrieval.WEAVIATE_ALIAS = os.getenv("WEAVIATE_ALIAS", "Main")
        
        logger.info(f"/save_config: Updated config keys: {changed_keys}")

    except Exception as e:
        logger.error(f"/save_config save error: {e}", exc_info=True)
        return jsonify(success=False, error="Failed to save configuration."), 500

    # 5. Background reload and pipeline init
    Thread(
        target=reload_and_maybe_reinit,
        args=(current_app.app_context(),),
        daemon=True
    ).start()

    # 6. Respond
    return jsonify(
        success=True,
        changed=True,
        updated=changed,
        message="Configuration saved — pipeline reload running in background.",
        config=validated.model_dump()
    ), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Return both the current config and the saved presets in one shot.
    """
    cfg_data = cfg.model_dump()
    cfg_data['presets'] = presets
    return jsonify(cfg_data), 200



@app.route('/start_ingestion', methods=['POST'])
def start_ingestion():
    """
    Triggers FULL document ingestion + centroid computation in one call, with preflight checks.
    """
    logger = current_app.logger

    # 1. Check that full-ingest components are available
    if not ingest_full_available or not DocumentProcessor or not PipelineConfig:
        logger.error("Full ingestion components unavailable.")
        return jsonify({'success': False, 'error': 'Full ingestion components unavailable.'}), 503
    if cfg is None:
        logger.error("Configuration system unavailable.")
        return jsonify({'success': False, 'error': 'Configuration system unavailable.'}), 500

    # 2. Get inputs with defaults
    data_folder = request.form.get('data_folder') or cfg.paths.DOCUMENT_DIR
    centroid_path = request.form.get('centroid_path') or cfg.paths.DOMAIN_CENTROID_PATH

    # 2.1 Sanity-check centroid_path
    if not isinstance(centroid_path, str) or not centroid_path.strip():
        msg = f"Invalid centroid_path: {centroid_path!r}"
        logger.error(msg)
        return jsonify({'success': False, 'error': msg}), 400

    # 3. Preflight: can we write the centroid file?
    save_dir = os.path.dirname(centroid_path) or '.'
    if not os.access(save_dir, os.W_OK):
        msg = f"Cannot write to directory: {save_dir}"
        logger.error(msg)
        return jsonify({'success': False, 'error': msg}), 500

    # 4. Preflight: data folder exists
    if not Path(data_folder).is_dir():
        msg = f"Data folder not found: {data_folder}"
        logger.error(msg)
        return jsonify({'success': False, 'error': msg}), 404

    # 5. Initialize Weaviate client
    weaviate_client = None
    try:
        logger.info("Creating fresh Weaviate client from PipelineConfig...")
        weaviate_client = PipelineConfig.get_client()
        if not weaviate_client or not weaviate_client.is_ready():
            raise ConnectionError("Weaviate client not ready.")

        # 6. Full ingestion
        doc_dir = Path(data_folder).resolve()
        logger.info(f"Starting full ingestion for directory: {doc_dir}")
        processor = DocumentProcessor(data_dir=str(doc_dir), client=weaviate_client)
        processor.execute()
        stats = getattr(processor, 'get_stats', lambda: {})()
        logger.info(f"Full ingestion complete. Stats: {stats}")

        # 7. Compute & save centroid (respect mode)
        mode = cfg.ingestion.CENTROID_UPDATE_MODE
        force_recalc = (mode == "always")
        logger.info(f"Centroid update mode: {mode} -> force={force_recalc}")

        result = calculate_and_save_centroid(
            weaviate_client,
            cfg.retrieval.COLLECTION_NAME,
            cfg.retrieval.WEAVIATE_ALIAS,
            os.path.dirname(centroid_path),
            force=force_recalc
        )

        if not result.get("ok"):
            err = result.get("error", "Centroid computation failed.")
            logger.error(f"Centroid error: {err}")
            return jsonify({
                'success': False,
                'error': err,
                'stats': stats
            }), 400

        logger.info(f"Centroid processed successfully. Path: {result.get('path')}")

        return jsonify({
            'success': True,
            'message': f"Full ingestion complete (centroid mode: {mode}).",
            'stats': stats,
            'centroid': result
        }), 200

    except FileNotFoundError as fnf_e:
        logger.error(f"Ingestion failed: {fnf_e}")
        return jsonify({'success': False, 'error': str(fnf_e)}), 404

    except (ConnectionError, WeaviateConnectionError) as conn_e:
        logger.error(f"Weaviate connection failed: {conn_e}")
        return jsonify({'success': False, 'error': str(conn_e)}), 503

    except Exception as e:
        logger.error(f"Full ingestion error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

    finally:
        if weaviate_client and hasattr(weaviate_client, 'close'):
            try:
                weaviate_client.close()
                logger.info("Weaviate client closed after ingestion.")
            except Exception as close_e:
                logger.error(f"Error closing Weaviate client: {close_e}")



@app.route("/ingest_block", methods=["POST"])
def ingest_block_route():
    try:
        # resolve paths
        centroid_path = request.form.get('centroid_path', cfg.paths.DOMAIN_CENTROID_PATH)
        document_dir = cfg.paths.DOCUMENT_DIR

        # ─── 1) BOOTSTRAP: if no centroid, do one ingest pass + force-create it ───
        if not centroid_exists(centroid_path):
            app.logger.warning(f"Centroid not found at {centroid_path}. Auto-creating…")
            init_result = run_incremental_ingestion(document_dir)
            if not isinstance(init_result, dict) or "new_vectors" not in init_result:
                msg = init_result.get("message", "Unknown ingestion error.")
                app.logger.error(f"Initial ingest failed: {msg}")
                return jsonify({"status": "error", "message": msg}), 500

            try:
                client = PipelineConfig.get_client()
                if not client or not client.is_ready():
                    raise ConnectionError("Weaviate client not ready.")
                result = calculate_and_save_centroid(
                    client,
                    cfg.retrieval.COLLECTION_NAME,
                    cfg.retrieval.WEAVIATE_ALIAS,
                    os.path.dirname(centroid_path),
                    force=True
                )
                if not result.get("ok"):
                    err = result.get("error", "Centroid computation failed.")
                    app.logger.error(f"Centroid error: {err}")
                    return jsonify({"status": "error", "message": err}), 400
            finally:
                try:
                    client.close()
                except:
                    pass

            app.logger.info("Fallback centroid created. Continuing with normal ingestion.")

        # ─── 2) NORMAL BLOCK INGESTION ───
        ingest_result = run_incremental_ingestion(document_dir)
        if not isinstance(ingest_result, dict) or "new_vectors" not in ingest_result:
            msg = ingest_result.get("message", "Unknown ingestion error.")
            app.logger.error(f"Incremental ingestion error: {msg}")
            return jsonify({"status": "error", "message": msg}), 500

        new_vectors = ingest_result["new_vectors"]
        all_vectors = ingest_result["all_vectors"]
        old_centroid = ingest_result.get("old_centroid")

        # pull user’s mode & threshold
        mode = request.form.get('centroid_update_mode', 'auto')
        try:
            raw_thr = float(request.form.get('centroid_auto_threshold') or cfg.ingestion.CENTROID_AUTO_THRESHOLD)
        except ValueError:
            raw_thr = cfg.ingestion.CENTROID_AUTO_THRESHOLD
        threshold = (raw_thr / 100.0) if raw_thr > 1 else raw_thr

        # decide if we recalc
        should_run = False
        if mode == 'always':
            should_run = True
        elif mode == 'auto':
            if old_centroid is None:
                should_run = True
            elif should_recalculate_centroid(
                new_vectors, all_vectors, old_centroid,
                threshold=threshold,
                diversity_threshold=cfg.ingestion.CENTROID_DIVERSITY_THRESHOLD
            ):
                should_run = True

        # recalc & save if needed
        centroid_result = None
        if should_run:
            app.logger.info("Recalculating centroid (mode=%s)…", mode)
            try:
                client = PipelineConfig.get_client()
                if not client or not client.is_ready():
                    raise ConnectionError("Weaviate client not ready.")
                centroid_result = calculate_and_save_centroid(
                    client,
                    cfg.retrieval.COLLECTION_NAME,
                    cfg.retrieval.WEAVIATE_ALIAS,
                    os.path.dirname(centroid_path),
                    force=True
                )
                if not centroid_result.get("ok"):
                    err = centroid_result.get("error", "Centroid computation failed.")
                    app.logger.error(f"Centroid error: {err}")
                    return jsonify({
                        "status": "error",
                        "message": err
                    }), 400
            finally:
                try:
                    client.close()
                except:
                    pass

        return jsonify({
            "status": "ok",
            "new_vectors": new_vectors,
            "all_vectors": all_vectors,
            "old_centroid": old_centroid,
            "centroid_updated": should_run,
            "centroid": centroid_result if centroid_result else None,
            "message": f"Ingested {len(new_vectors)} new vectors; centroid {'updated' if should_run else 'unchanged'}."
        }), 200

    except Exception as e:
        app.logger.error(f"/ingest_block error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc().splitlines()[-5:]
        }), 500




# --- Chat History Routes ---
# UPDATED: Use JSON for save/delete actions

@app.route('/run_keyword_builder', methods=['POST'])
def run_keyword_builder():
    """Runs the keyword_builder_v3.py script with provided parameters."""
    try:
        # 1) Validate & parse JSON payload
        payload = KeywordBuilderRequest.parse_obj(request.get_json() or {})

        # 2) Docker & config availability
        if not docker_available:
            return jsonify(success=False, error="Docker client unavailable."), 503
        if cfg is None:
            return jsonify({"error": "Config unavailable."}), 503

        # 3) Weaviate connectivity check
        port = cfg.retrieval.WEAVIATE_HTTP_PORT
        if not check_weaviate_connection(port):
            return jsonify(
                success=False,
                error=f"Cannot connect to Weaviate on port {port}. Ensure service is running."
            ), 503

        # 4) Build the command
        cmd = [
            sys.executable,
            "keywords_builder_v3.py",
            "--collection", cfg.retrieval.COLLECTION_NAME,
            "--keybert_model", payload.keybert_model,
            "--top_n_per_doc", str(payload.top_n_per_doc),
            "--final_top_n",   str(payload.final_top_n),
            "--min_doc_freq",  str(payload.min_doc_freq),
            "--diversity",     str(payload.extraction_diversity),
        ]
        if payload.no_pos_filter:
            cmd.append("--no_pos_filter")

        logger.info(f"Running keyword builder: {' '.join(cmd)}")

        # 5) Launch subprocess with timeout
        timed_out = False
        def kill_proc():
            nonlocal timed_out
            timed_out = True
            process.kill()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ.copy()
        )
        timer = Timer(payload.timeout, kill_proc)
        try:
            timer.start()
            stdout, stderr = process.communicate()
        finally:
            timer.cancel()

        if timed_out:
            logger.error(f"Timed out after {payload.timeout}s")
            return jsonify(
                success=False,
                error=f"Process timed out after {payload.timeout} seconds",
                details="Extraction took too long."
            ), 504

        if process.returncode != 0:
            logger.error(f"Builder failed: {stderr}")
            last_lines = "\n".join(stderr.strip().splitlines()[-3:])
            return jsonify(
                success=False,
                error=f"Exit code {process.returncode}",
                details=last_lines or "Unknown error",
                full_error=stderr
            ), 500

        # 6) Parse keywords
        keywords, in_section = [], False
        for line in stdout.splitlines():
            if "--- Top" in line and "Filtered Domain Keywords" in line:
                in_section = True
                continue
            if in_section and line.startswith("-----------------------------------------------------------------------------"):
                break
            if in_section and " : " in line:
                term, score = line.split(" : ", 1)
                try:
                    keywords.append({"term": term.strip(), "score": float(score.strip())})
                except ValueError:
                    logger.warning(f"Can't parse score: {line}")

        if not keywords:
            logger.warning("No keywords extracted")
            info = stdout or stderr
            return jsonify(
                success=False,
                error="No keywords extracted",
                details=info
            ), 500

        logger.info(f"Extracted {len(keywords)} keywords")

        # 7) Caching (if enabled)
        if getattr(cfg.security, 'CACHEENABLED', False):
            cache_key = f"keywords_{cfg.retrieval.COLLECTION_NAME}_{int(time.time())}"
            try:
                cache_results(cache_key, {
                    "keywords": keywords,
                    "parameters": payload.dict(by_alias=True),
                    "timestamp": time.time()
                })
                logger.info(f"Cached key: {cache_key}")
            except Exception as e:
                logger.warning(f"Cache failed: {e}")

        # 8) Save debug output
        try:
            out_dir = os.path.join(os.path.dirname(__file__), "data", "outputs")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"keyword_builder_output_{ts}.txt")
            with open(path, 'w') as f:
                f.write(f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
            logger.info(f"Output saved to {path}")
        except Exception as e:
            logger.warning(f"Save output failed: {e}")

        # 9) Sync auto-domain keywords
        requests.post(
            url_for('update_auto_domain_keywords', _external=True),
            json={"keywords": [kw["term"] for kw in keywords], "target_field": "AUTO_DOMAIN_KEYWORDS"},
            timeout=10
        )

        # 10) Return success
        return jsonify(
            success=True,
            keywords=keywords,
            count=len(keywords),
            parameters=payload.dict(by_alias=True),
            message=f"Successfully extracted {len(keywords)} keywords"
        )

    except Exception as e:
        logger.error(f"Unexpected error in run_keyword_builder: {e}", exc_info=True)
        return jsonify(success=False, error=str(e)), 500


def check_weaviate_connection(port=8080):
    """Check if Weaviate is accessible on the specified port."""
    try:
               
        url = f"http://localhost:{port}/v1/.well-known/ready"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except RequestException as e:
        app.logger.error(f"Weaviate connection check failed: {str(e)}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error checking Weaviate connection: {str(e)}")
        return False


def cache_results(key, data):
    """Cache the results in memory or persistent storage."""
    # This is a placeholder function - implement according to your caching strategy
    # For example, you might use Redis, a database, or a simple file-based cache
    
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{key}.json")
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return True

import atexit

def cleanup_resources():
    # Close pipeline retriever on the actual pipeline instance
    try:
        from pipeline import PIPELINE_INSTANCE
        inst = PIPELINE_INSTANCE
    except ImportError:
        inst = None

    if inst and hasattr(inst.retriever, 'close'):
        try:
            inst.retriever.close()
            logger.info("Closed pipeline retriever client")
        except Exception as e:
            logger.error(f"Error closing pipeline retriever: {e}", exc_info=True)

    # Close any other Weaviate clients
    try:
        logger.info("Attempting to close any remaining Weaviate connections")
        import weaviate

        for attr in dir(weaviate):
            if attr.endswith("Client"):
                client_cls = getattr(weaviate, attr, None)
                instances = getattr(client_cls, "_instances", None)
                if instances:
                    for client in list(instances):
                        close_method = getattr(client, "close", None)
                        if callable(close_method):
                            try:
                                close_method()
                                logger.info(f"Closed {attr} instance")
                            except Exception as e:
                                logger.error(f"Error closing {attr} instance: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during additional Weaviate connection cleanup: {e}", exc_info=True)

atexit.register(cleanup_resources)


   
@app.route("/centroid_stats")
def centroid_stats():
    collection = request.args.get("collection", "").strip()
    instance = request.args.get("instance", "").strip()

    if not collection or not instance:
        return jsonify({"success": False, "error": "Missing collection or instance parameter"}), 400

    try:
        cm = CentroidManager(
            instance_alias=instance,
            collection_name=collection,
            base_path=cfg.paths.CENTROID_DIR
        )
        centroid = cm.get_centroid()
    except Exception as e:
        app.logger.error(f"[centroid_stats] Failed to initialize CentroidManager: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to load centroid"}), 500

    if centroid is None:
        return jsonify({"success": False, "stats": {"meta": None}})

    meta = cm.get_metadata()
    expected_keys = {"Mean", "Std Dev", "Min", "Max", "L2 Norm (Magnitude)"}

    if not isinstance(meta, dict) or not expected_keys.issubset(meta.keys()):
        meta = get_centroid_stats(centroid)
        cm.save_metadata(meta)

    return jsonify({
        "success": True,
        "stats": {
            "meta": meta
        }
    })




@app.route('/centroid_histogram.png')
def centroid_histogram():
    current_app.logger.info("-> /centroid_histogram.png endpoint called")

    collection_name = request.args.get("collection", "").strip()
    instance_alias = request.args.get("instance", "").strip()

    if not collection_name or not instance_alias:
        current_app.logger.warning("-> Missing 'collection' or 'instance' query parameter.")
        placeholder = os.path.join(current_app.static_folder, "placeholder.png")
        return send_file(placeholder, mimetype="image/png")

    current_app.logger.info(f"-> Using collection: {collection_name} (instance: {instance_alias})")

    try:
        cm = CentroidManager(
            instance_alias=instance_alias,
            collection_name=collection_name,
            base_path=cfg.paths.CENTROID_DIR
        )
        centroid = cm.get_centroid()
    except Exception as e:
        current_app.logger.error(f"Failed to load centroid: {e}", exc_info=True)
        placeholder = os.path.join(current_app.static_folder, "placeholder.png")
        return send_file(placeholder, mimetype="image/png")

    if centroid is None:
        current_app.logger.warning("-> No centroid found, serving placeholder.")
        placeholder = os.path.join(current_app.static_folder, "placeholder.png")
        return send_file(placeholder, mimetype="image/png")

    current_app.logger.info(f"-> Loaded centroid of shape {getattr(centroid, 'shape', None)}")

    try:
        fig, ax = plt.subplots()
        ax.hist(centroid, bins=50)
        ax.set_title(f"Centroid Distribution — {instance_alias}_{collection_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype="image/png")
    except Exception as e:
        current_app.logger.error(f"Failed to generate histogram: {e}", exc_info=True)
        placeholder = os.path.join(current_app.static_folder, "placeholder.png")
        return send_file(placeholder, mimetype="image/png")




################## @app.route from here on
   
@app.route('/update_config_keywords', methods=['POST'])
def update_config_keywords():
    """Updates the config with extracted keywords."""
    logger = app.logger

    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503

    try:
        data = request.get_json()
        keywords = data.get('keywords', [])

        if not keywords:
            return jsonify({"success": False, "error": "No keywords provided"}), 400

        updates = {
            'env': {
                'AUTO_DOMAIN_KEYWORDS': keywords
            }
        }

        changed = cfg.update_and_save(updates)

        if changed:
            try:
                cfg.reload()
                cfg.retrieval.WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
                cfg.retrieval.WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8092"))
                cfg.retrieval.WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "51003"))
                cfg.retrieval.WEAVIATE_ALIAS = os.getenv("WEAVIATE_ALIAS", "Main")
                logger.info(f"Reloaded cfg after keyword update. {len(keywords)} auto domain keywords set.")
            except Exception as reload_err:
                logger.error(f"Failed to reload config: {reload_err}", exc_info=True)
                return jsonify({"success": False, "error": "Reload failed after config update."}), 500

            initialize_pipeline(app.app_context())

            return jsonify({
                "success": True,
                "message": f"Updated configuration with {len(keywords)} keywords"
            })

        else:
            return jsonify({
                "success": True,
                "message": "No configuration changes were needed"
            })

    except Exception as e:
        logger.error(f"Error updating config with keywords: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500



@app.route('/save_chat', methods=['POST'])
def save_chat():
    """Saves chat history. Expects JSON { name: "...", history: [...] }."""
    #logger = app.logger
    if 'chat_history' not in session or not session['chat_history']:
        return jsonify({"success": False, "error": "No chat history in session to save."}), 400
    try:
        data = request.get_json()
        chat_name = data.get('name', '').strip()
        # Use history from request OR session? Let's use session for now.
        history_to_save = session['chat_history'] # Or data.get('history') if JS sends it

        if not chat_name: return jsonify({"success": False, "error": "Chat name required."}), 400
        if not history_to_save: return jsonify({"success": False, "error": "No history data provided."}), 400

        new_chat = SavedChat(name=chat_name, history=history_to_save)
        db.session.add(new_chat)
        db.session.commit()
        logger.info(f"Saved chat '{chat_name}' with ID {new_chat.id}")
        return jsonify({"success": True, "id": new_chat.id, "name": new_chat.name, "message": f"Chat '{chat_name}' saved."})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving chat: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500


@app.route('/load_chat/<int:chat_id>', methods=['GET'])
def load_chat(chat_id):
    """Loads a specific chat history into the current session."""
    #logger = app.logger
    try:
        chat = db.session.get(SavedChat, chat_id)
        if chat:
            session['chat_history'] = chat.history
            logger.info(f"Loaded chat '{chat.name}' (ID: {chat_id}) into session.")
            return jsonify({"success": True, "id": chat.id, "name": chat.name, "history": chat.history})
        else:
            return jsonify({"success": False, "error": "Chat not found."}), 404
    except Exception as e:
        logger.error(f"Error loading chat ID {chat_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500



@app.route('/list_chats', methods=['GET'])
def list_chats():
    """Lists all saved chats (ID, Name, Timestamp)."""
    #logger = app.logger
    try:
        chats = SavedChat.query.order_by(SavedChat.created_at.desc()).all()
        # Include timestamp in response
        chat_list = [{"id": c.id, "name": c.name, "timestamp": c.created_at.isoformat()} for c in chats]
        return jsonify(chat_list)
    except Exception as e:
        logger.error(f"Error listing chats: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {e}"}), 500


@app.route('/delete_chat/<int:chat_id>', methods=['DELETE']) # Use DELETE method
def delete_chat(chat_id):
    """Deletes a saved chat."""
    #logger = app.logger
    try:
        chat = db.session.get(SavedChat, chat_id)
        if chat:
            db.session.delete(chat)
            db.session.commit()
            logger.info(f"Deleted saved chat '{chat.name}' (ID: {chat_id}).")
            return jsonify({"success": True, "message": "Chat deleted."})
        else:
            return jsonify({"success": False, "error": "Chat not found."}), 404
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting chat ID {chat_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500

# UPDATED: Expects JSON { instance_name: "..." }
@app.route('/remove_weaviate_instance', methods=['POST']) # Or DELETE if you prefer
def remove_weaviate_instance():
    if not docker_available:
        return jsonify({"success": False, "error": "Docker client unavailable."}), 503
    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503

    logger = app.logger
    data = request.get_json()
    instance_name = data.get('instance_name')

    if not instance_name:
        logger.warning("API /remove_weaviate_instance: No instance name provided.")
        return jsonify({"success": False, "error": "Instance name required."}), 400

    logger.info(f"API /remove_weaviate_instance: Request received for '{instance_name}'.")

    instance_state = load_weaviate_state()
    instance_to_delete = instance_state.get(instance_name)

    if not instance_to_delete:
        logger.warning(f"API /remove_weaviate_instance: Instance '{instance_name}' not found in state file.")
        # Return 404 even if container might exist, state file is the source of truth for managed instances
        return jsonify({"success": False, "error": f"Instance '{instance_name}' not found."}), 404

    container_id = instance_to_delete.get('container_id')
    instance_http_port = instance_to_delete.get('http_port')
    instance_host = instance_to_delete.get('host', 'localhost') # Assume localhost if host is missing

    # --- Check if this is the currently active instance ---
    is_active_instance = False
    try:
        # Compare ports (ensure types match)
        if str(instance_host) == str(cfg.retrieval.WEAVIATE_HOST) and \
           str(instance_http_port) == str(cfg.retrieval.WEAVIATE_HTTP_PORT):
            is_active_instance = True
            logger.info(f"Instance '{instance_name}' is the currently active instance.")
    except Exception as e:
        logger.error(f"Error comparing ports while checking active status for deletion: {e}")
        # Proceed with caution, maybe don't attempt auto-switch

    fallback_activated = False
    fallback_message = ""

    # --- Stop and Remove Container ---
    if container_id:
        try:
            container = docker_client.containers.get(container_id)
            logger.info(f"Stopping container {container_id} for instance '{instance_name}'...")
            container.stop()
            logger.info(f"Removing container {container_id}...")
            container.remove()
            logger.info(f"Container {container_id} removed.")
            # Optionally remove associated volume if needed (use with caution!)
            # volume_name = instance_to_delete.get('volume_name')
            # if volume_name: try: docker_client.volumes.get(volume_name).remove() except DockerNotFound: pass
        except DockerNotFound:
            logger.warning(f"Container {container_id} for '{instance_name}' not found, maybe already removed.")
        except DockerAPIError as e:
            logger.error(f"Docker error stopping/removing container {container_id}: {e}")
            # Decide if you should still proceed with removing from state
            # Maybe return error here? For now, we proceed to remove from state.
        except Exception as e:
            logger.error(f"Unexpected error managing container {container_id}: {e}")
            # Decide if you should still proceed with removing from state

    # --- Update State File ---
    if instance_name in instance_state:
        del instance_state[instance_name]
        save_weaviate_state(instance_state)
        logger.info(f"Instance '{instance_name}' removed from state file.")
    else:
        # This case should be rare if we found it earlier, but good to log
        logger.warning(f"Instance '{instance_name}' was already missing from state before saving.")


    # --- Handle Fallback Activation if Active Instance was Deleted ---
    if is_active_instance:
        logger.info(f"Active instance '{instance_name}' deleted. Attempting to activate fallback...")
        remaining_instances = instance_state # State is already updated

        new_active_instance_name = None
        new_active_details = None

        if remaining_instances:
            # Simple strategy: activate the first remaining instance alphabetically
            first_remaining_name = sorted(remaining_instances.keys())[0]
            new_active_instance_name = first_remaining_name
            new_active_details = remaining_instances[new_active_instance_name]
            logger.info(f"Selected '{new_active_instance_name}' as fallback active instance.")
        else:
            # No instances left, potentially revert to a hardcoded default?
            # Or leave config pointing to the deleted one (will cause errors later)?
            # Let's revert to reasonable defaults if no instances left.
            logger.warning("No remaining instances to activate. Reverting config to default ports.")
            new_active_details = {
                'host': 'localhost',
                'http_port': 8080, # Or your chosen default
                'grpc_port': 50051  # Or your chosen default
            }
            new_active_instance_name = "Default (no instances left)"


        if new_active_details:
            try:
                # Prepare updates dictionary for config save
                config_updates = {
                    "retrieval": {
                        "WEAVIATE_HOST": new_active_details.get('host', 'localhost'),
                        "WEAVIATE_HTTP_PORT": int(new_active_details.get('http_port', 8080)),
                        "WEAVIATE_GRPC_PORT": int(new_active_details.get('grpc_port', 50051))
                    }
                }
                # Update config in memory AND save to YAML
                cfg.update_and_save(config_updates)
                try:
                    cfg.reload()
                    cfg.retrieval.WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", cfg.retrieval.WEAVIATE_HOST)
                    cfg.retrieval.WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", cfg.retrieval.WEAVIATE_HTTP_PORT))
                    cfg.retrieval.WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", cfg.retrieval.WEAVIATE_GRPC_PORT))
                    cfg.retrieval.WEAVIATE_ALIAS = os.getenv("WEAVIATE_ALIAS", cfg.retrieval.WEAVIATE_ALIAS)
                    app.logger.info(f"Reloaded configuration after update. New host: {cfg.retrieval.WEAVIATE_HOST}, HTTP: {cfg.retrieval.WEAVIATE_HTTP_PORT}, gRPC: {cfg.retrieval.WEAVIATE_GRPC_PORT}")
                except Exception as reload_err:
                    app.logger.error(f"Failed to reload config after saving instance selection: {reload_err}", exc_info=True)
                    return jsonify({"error": "Configuration reload failed after update."}), 500
                logger.info(f"Configuration updated and saved to use fallback: {new_active_instance_name} ({config_updates['retrieval']})")

                # Re-initialize the pipeline to use the new configuration
                initialize_pipeline(app.app_context())
                if not pipeline:
                     logger.error(f"Pipeline failed re-initialization after activating fallback '{new_active_instance_name}'!")
                     fallback_message = f" Instance '{instance_name}' deleted, but pipeline failed to switch to fallback."
                     # Return error?
                     return jsonify({"success": False, "error": fallback_message}), 500
                else:
                    fallback_activated = True
                    fallback_message = f" Instance '{instance_name}' deleted. Pipeline now active on fallback: '{new_active_instance_name}'."

            except Exception as e:
                logger.error(f"Failed to update config or re-initialize pipeline for fallback instance: {e}", exc_info=True)
                fallback_message = f" Instance '{instance_name}' deleted, but failed to activate fallback automatically: {e}"
                # Return error? The original instance IS deleted. Maybe return success=True but with error message?
                # Let's return success but indicate the activation issue.
                return jsonify({"success": True, "message": fallback_message}), 200 # 200 OK but with message indicating fallback issue


    # --- Final Response ---
    final_message = f"Instance '{instance_name}' removed successfully."
    if fallback_message:
        final_message += fallback_message

    return jsonify({"success": True, "message": final_message}), 200

# --- END MODIFIED /remove_weaviate_instance ---


# --- Multi-Weaviate Routes ---
# UPDATED: Ensure JSON handling

@app.route("/create_weaviate_instance", methods=["POST"])
def create_weaviate_instance():
    #logger = app.logger
    if not docker_available: return jsonify({"error": "Docker client unavailable."}), 503
    try:
        data = request.get_json()
        instance_name_req = data.get('instance_name', '').strip()
    except Exception as req_e: return jsonify({"error": "Invalid request data."}), 400

    if not instance_name_req or not re.match(r'^[a-zA-Z0-9_-]+$', instance_name_req) or len(instance_name_req) > 50:
        return jsonify({"error": "Invalid instance name."}), 400
    
    container_name = f"rag_weaviate_{instance_name_req}"
    volume_name = f"weaviate_data_{instance_name_req}"
    instance_state = load_weaviate_state()
    
    if container_name in instance_state or instance_name_req in instance_state:
        return jsonify({"error": f"Instance '{instance_name_req}' already exists."}), 409
    # --- >>> INSERT DOCKER CHECK BLOCK HERE <<< ---
    try:
        logger.debug(f"Checking Docker for existing container named '{container_name}'...")
        # Use list with exact name filter, include stopped containers (all=True)
        existing_containers = docker_client.containers.list(all=True, filters={'name': container_name})
        if existing_containers:
            # Name is already in use in Docker
            existing_id = existing_containers[0].id[:12] # Get short ID for message
            logger.error(f"Docker conflict: Container name '{container_name}' is already in use by container ID {existing_id}.")
            return jsonify({
                "error": f"Container name '{container_name}' is already in use by Docker (Container ID: {existing_id}). Please remove the existing container via Docker CLI ('docker rm {container_name}') or choose a different instance name."
            }), 409 # Conflict (Resource already exists)
        else:
            logger.debug(f"Docker check passed: Container name '{container_name}' is available.")
    except DockerAPIError as docker_check_err:
        # Handle errors during the Docker check itself
        logger.error(f"Error checking Docker for existing container '{container_name}': {docker_check_err}")
        return jsonify({"error": f"Error communicating with Docker daemon: {docker_check_err.explanation}"}), 500
    except Exception as check_e:
        logger.error(f"Unexpected error during Docker container check: {check_e}", exc_info=True)
        return jsonify({"error": f"Unexpected error checking Docker status: {str(check_e)}"}), 500
    # --- >>> END INSERTED DOCKER CHECK BLOCK <<< ---
    try:
        next_http_port, next_grpc_port = get_next_ports(instance_state)
        logger.info(f"Starting Weaviate '{container_name}' on HTTP:{next_http_port}, gRPC:{next_grpc_port}")
        
        # Create a Docker-managed volume instead of a bind mount
        logger.info(f"Creating Docker volume '{volume_name}'")
        try:
            volume = docker_client.volumes.create(name=volume_name)
            logger.info(f"Created Docker volume: {volume.name}")
        except DockerAPIError as vol_err:
            logger.error(f"Failed to create Docker volume '{volume_name}': {vol_err}")
            return jsonify({"error": f"Volume creation error: {vol_err.explanation}"}), 500
        
        container_persist_path = "/var/lib/weaviate"
        
        container_config = {
            "image": WEAVIATE_IMAGE, 
            "name": container_name,
            "ports": {'8080/tcp': next_http_port, '50051/tcp': next_grpc_port},
            "environment": {
                "PERSISTENCE_DATA_PATH": container_persist_path, 
                "DEFAULT_VECTORIZER_MODULE": "none",
                "ENABLE_MODULES": "", 
                "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                "CLUSTER_HOSTNAME": f"node_{instance_name_req}", 
                "QUERY_DEFAULTS_LIMIT": "25", 
                "LOG_LEVEL": "info"
            },
            # Use the Docker-managed volume
            "volumes": {volume_name: {'bind': container_persist_path, 'mode': 'rw'}},
            "detach": True, 
            "restart_policy": {"Name": "unless-stopped"}
        }
        
        container = docker_client.containers.run(**container_config)
        
        state_key = instance_name_req
        instance_state[state_key] = {
            "status": "starting",
            "host": "localhost", 
            "http_port": next_http_port, 
            "grpc_port": next_grpc_port,
            "container_id": container.id, 
            "container_name": container_name,
            "volume_name": volume_name,  # Store the volume name in the state
            "created_at": datetime.now().isoformat()
        }
        
        save_weaviate_state(instance_state)
        logger.info(f"Started container '{container.name}' with volume '{volume_name}' for instance '{state_key}'")
        
        response_details = instance_state[state_key].copy()
        response_details["name"] = state_key
        return jsonify({
            "success": True, 
            "message": f"Instance '{state_key}' creating with Docker volume '{volume_name}'...", 
            "details": response_details
        }), 201
    
    except DockerAPIError as api_err: 
        # Clean up volume if container creation fails
        try:
            if 'volume' in locals() and volume:
                logger.info(f"Cleaning up volume '{volume.name}' after container creation failure")
                volume.remove(force=True)
        except Exception as vol_cleanup_err:
            logger.error(f"Failed to clean up volume after error: {vol_cleanup_err}")
            
        logger.error(f"Docker error creating '{instance_name_req}': {api_err}", exc_info=True)
        return jsonify({"error": f"Docker error: {api_err.explanation}"}), 500
    
    except Exception as e: 
        # Clean up volume if container creation fails
        try:
            if 'volume' in locals() and volume:
                logger.info(f"Cleaning up volume '{volume.name}' after container creation failure")
                volume.remove(force=True)
        except Exception as vol_cleanup_err:
            logger.error(f"Failed to clean up volume after error: {vol_cleanup_err}")
            
        logger.error(f"Error creating instance '{instance_name_req}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to create instance: {e}"}), 500



@app.route("/list_weaviate_instances", methods=["GET"])
def list_weaviate_instances():
    """Returns list of managed Weaviate instances with active instance properly labeled."""
    #logger = app.logger

    try:
        # ─── 1) Load default-from-config ─────────────────────────────────────────
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        cfg_host      = cfg_yaml.get("retrieval", {}).get("WEAVIATE_HOST", "localhost")
        default_http  = int(cfg_yaml.get("retrieval", {}).get("WEAVIATE_HTTP_PORT", 8080) or 8080)
        default_grpc  = int(cfg_yaml.get("retrieval", {}).get("WEAVIATE_GRPC_PORT", 50051) or 50051)
        is_default_active = (
            cfg
            and cfg.retrieval.WEAVIATE_HOST == cfg_host
            and cfg.retrieval.WEAVIATE_HTTP_PORT == default_http
        )
        default_instance = {
            "name":       "Active (from current config)",
            "host":       cfg_host,
            "http_port":  default_http,
            "grpc_port":  default_grpc,
            "status":     "config_default",
            "container_id":   None,
            "container_name": None,
            "active":     is_default_active
        }

        # ─── 2) Discover running containers ───────────────────────────────────────
        running = []
        if docker_available:
            running = docker_client.containers.list(filters={"ancestor": "weaviate"})
        # map port‐strings so we can filter default
        running_ports = {
            c.attrs["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"]
            for c in running
            if c.attrs.get("NetworkSettings", {}).get("Ports", {}).get("8080/tcp")
        }

        # ─── 3) Load your saved presets / managed instances ────────────────────────
        managed = []
        state = load_weaviate_state()
        for name, details in state.items():
            cid = details.get("container_id")
            status = details.get("status", "unknown")
            # if the container still exists, get its real status
            if docker_available and cid:
                try:
                    status = docker_client.containers.get(cid).status
                except Exception:
                    status = "not_found"
            managed.append({
                "name":           name,
                "host":           details.get("host", cfg_host),
                "http_port":      details.get("http_port", default_http),
                "grpc_port":      details.get("grpc_port", default_grpc),
                "status":         status,
                "container_id":   cid,
                "container_name": details.get("container_name"),
                "active":         (
                    cfg
                    and cfg.retrieval.WEAVIATE_HOST == details.get("host")
                    and cfg.retrieval.WEAVIATE_HTTP_PORT == details.get("http_port")
                )
            })

        # ─── 4) Auto-add any OTHER running containers not in your state ───────────
        known_cids = {inst["container_id"] for inst in managed if inst["container_id"]}
        for c in running:
            if c.id in known_cids:
                continue
            # grab its port
            try:
                port_str = c.attrs["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"]
                port = int(port_str)
            except Exception:
                continue
            managed.append({
                "name":           c.name,
                "host":           cfg_host,
                "http_port":      port,
                "grpc_port":      default_grpc,
                "status":         c.status,
                "container_id":   c.id,
                "container_name": c.name,
                "active":         (cfg and cfg.retrieval.WEAVIATE_HTTP_PORT == port)
            })

        # ─── 5) Combine + filter out “default” when ghost ─────────────────────────
        final_list = []
        # only include the default entry if its port is actually running
        if str(default_http) in running_ports:
            # if one of our managed instances matches its port, rename it “Active: instance – X”
            match = next(
                (i for i in managed
                 if i["http_port"] == default_http and i["host"] == cfg_host),
                None
            )
            if match:
                primary = default_instance.copy()
                primary["name"]   = f'Active: instance - "{match["name"]}"'
                primary["active"] = True
                # drop that managed entry so we don’t duplicate
                rest = [i for i in managed if i is not match]
                final_list = [primary] + rest
            else:
                # no match -> show default as-is
                final_list = [default_instance] + managed
        else:
            # default port isn’t live -> skip default entirely
            final_list = managed

        # ─── 6) Sort & return ──────────────────────────────────────────────────────
        final_list.sort(key=lambda i: (not i.get("active", False), i.get("name", "")))
        return jsonify(final_list)

    except Exception as e:
        logger.error("Error listing instances", exc_info=True)
        return jsonify({"error": "Failed to list instances"}), 500


@app.route("/select_weaviate_instance", methods=["POST"])
def select_weaviate_instance():
    """Activate a Weaviate instance, persist it, and rebuild the pipeline."""
    data = request.get_json(silent=True) or {}
    name = data.get("instance_name") or data.get("name")
    if not name:
        return jsonify({"error": "Instance name required."}), 400
    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503

    # 1) Determine host/port
    try:
        if name == "Default (from config)":
            with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            host = raw.get("retrieval", {}).get("WEAVIATE_HOST", "localhost")
            http_port = int(raw.get("retrieval", {}).get("WEAVIATE_HTTP_PORT", 8080) or 8080)
            grpc_port = int(raw.get("retrieval", {}).get("WEAVIATE_GRPC_PORT", 50051) or 50051)
            alias = raw.get("retrieval", {}).get("WEAVIATE_ALIAS", "Default")
            app.logger.info(f"Selecting default-from-config -> {host}:{http_port}")
        else:
            state = load_weaviate_state()
            details = state.get(name)
            if not details:
                return jsonify({"error": f"Instance '{name}' not found."}), 404
            host = details.get("host")
            http_port = int(details.get("http_port", -1))
            grpc_port = int(details.get("grpc_port", -1))
            alias = name  # always use the selected name
            if not host or http_port < 0 or grpc_port < 0:
                return jsonify({"error": "Invalid connection details."}), 400
            app.logger.info(f"Selecting managed instance '{name}' -> {host}:{http_port}")
    except Exception as e:
        app.logger.error("Error reading instance details", exc_info=True)
        return jsonify({"error": "Failed to read instance details."}), 500

    # 2) Persist to config
    try:
        new_retrieval_config = {
            "WEAVIATE_HOST": host,
            "WEAVIATE_HTTP_PORT": http_port,
            "WEAVIATE_GRPC_PORT": grpc_port,
            "WEAVIATE_ALIAS": alias
        }
        app.logger.info(f"Persisting retrieval config: {new_retrieval_config}")
        cfg.update_and_save({"retrieval": new_retrieval_config})
    except Exception as e:
        app.logger.error("Failed to save config", exc_info=True)
        return jsonify({"error": "Could not persist configuration."}), 500

    # 3) Manage Docker containers if applicable
    if docker_available:
        try:
            state = load_weaviate_state()
            for nm, det in state.items():
                if nm != name and det.get("container_id"):
                    ctr = docker_client.containers.get(det["container_id"])
                    if ctr.status == "running":
                        app.logger.info(f"Stopping '{nm}'...")
                        ctr.stop(timeout=30)
                        state[nm]["status"] = "exited"
            if name != "Default (from config)" and state[name].get("container_id"):
                ctr = docker_client.containers.get(state[name]["container_id"])
                if ctr.status != "running":
                    app.logger.info(f"Starting '{name}'...")
                    ctr.start()
                    state[name]["status"] = "running"
            save_weaviate_state(state)
        except Exception as e:
            app.logger.error("Error controlling Docker instances", exc_info=True)

    app.logger.warning(f"[DEBUG] Probing URL: http://{host}:{http_port}/v1/.well-known/openid-configuration")

    # 4) Wait for REST to be ready
    if not _wait_for_weaviate_rest(host, http_port, retries=15, delay=0.5):
        app.logger.error(f"Weaviate REST not up at {host}:{http_port}")
        return jsonify({"error": "Instance failed to start in time."}), 502

    # 5) Re-init pipeline (IMPORTANT: no reload)
    try:
        from pipeline import init_pipeline_once
        inst = init_pipeline_once(force=True)
        app.logger.info(f"Pipeline re-initialized with alias: {inst.cfg.retrieval.WEAVIATE_ALIAS}")
    except Exception as e:
        app.logger.error("Pipeline re-init failed", exc_info=True)
        return jsonify({"error": "Pipeline re-initialization failed."}), 500

    # 6) Verify connectivity
    retriever = getattr(inst, "retriever", None)
    client = getattr(retriever, "weaviate_client", None)
    if not client:
        app.logger.error("Pipeline has no Weaviate client.")
        return jsonify({"error": "Pipeline missing client reference."}), 500

    if not client.is_ready():
        app.logger.error("Weaviate client reports not ready after switch.")
        return jsonify({"error": "Pipeline cannot connect to new instance."}), 500

    # Extra: List collections to confirm
    try:
        col_list = client.collections.list_all()
        app.logger.info(f"Connected collections: {col_list}")
    except Exception as e:
        app.logger.error("Collection listing failed after switch.", exc_info=True)
        return jsonify({"error": "Pipeline cannot list collections on new instance."}), 500

    app.logger.info(f"Pipeline successfully connected to '{name}'.")
    return jsonify({
        "success": True,
        "message": f"Instance '{name}' is now active.",
        "active_host": host,
        "active_http_port": http_port
    }), 200

@app.route('/list_presets', methods=['GET'])
def list_presets_api():
    """Return all saved presets as JSON for the frontend dropdown."""
    try:
        data = load_presets()  # loads from PRESETS_FILE and populates `presets` :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        return jsonify(data), 200
    except Exception as e:
        app.logger.error(f"API /list_presets error: {e}", exc_info=True)
        # Return empty object so the frontend handles it gracefully
        return jsonify({}), 500


@app.route("/list_routes")
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": sorted([m for m in rule.methods if m not in ('HEAD', 'OPTIONS')]), # Filter methods
            "path": str(rule)
        })
    return jsonify({"routes": sorted(routes, key=lambda x: x["path"])})


@app.route('/get_auto_domain_keywords', methods=['GET'])
def get_auto_domain_keywords():
    keywords = []
    # 1. Try config first
    if hasattr(cfg, "env"):
        raw = getattr(cfg.env, "AUTO_DOMAIN_KEYWORDS", [])
        # Handle both formats: list of strings, or list with one comma-separated string
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], str) and ',' in raw[0]:
                # Split the single string into keywords
                keywords = [kw.strip() for kw in raw[0].split(",") if kw.strip()]
            else:
                # Already a list of keywords
                keywords = raw
        elif isinstance(raw, str):
            keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
    # 2. Fallback: try file if config is empty
    if not keywords:
        try:
            with open("auto_domain_keywords.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    keywords = [kw.strip() for kw in content.split(",") if kw.strip()]
        except Exception as e:
            logger.error(f"Failed to read auto_domain_keywords.txt: {e}")

    logger.info(f"Returning auto domain keywords: {keywords}")
    return jsonify({"success": bool(keywords), "keywords": keywords})

    
    # app.py (add after other imports)

centroid_manager = CentroidManager(
    instance_alias=cfg.retrieval.WEAVIATE_ALIAS,
    collection_name=cfg.retrieval.COLLECTION_NAME,
    base_path=cfg.paths.CENTROID_DIR
)
centroid = centroid_manager.get_centroid()

@app.route('/api/collections', methods=['GET'])
def list_collections():
    """
    Returns a JSON list of all Weaviate collections (i.e., class names).
    """
    client = None
    try:
        client = PipelineConfig.get_client()
        if not client or not client.is_live() or not client.is_ready():
            current_app.logger.error("Weaviate not ready in /api/collections")
            return jsonify({"error": "Weaviate service unavailable"}), 503

        # Try fast-path
        try:
            names = client.collections.list_all(simple=True)
            if isinstance(names, dict):
                names = list(names.keys())
            elif not isinstance(names, list):
                raise TypeError(f"Unexpected format from simple=True: {type(names)}")
        except Exception as e:
            current_app.logger.warning(f"Simple fetch failed: {e}, falling back to full list.")
            configs = client.collections.list_all()
            names = [c.get("class") or c.get("name") for c in configs if isinstance(c, dict)]

        return jsonify({"collections": names}), 200

    except Exception as e:
        current_app.logger.error(f"API /api/collections error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

    

VALID_NAME = re.compile(r"^[A-Za-z0-9_]+$")

@app.route('/api/centroid', methods=['GET', 'POST'])
def centroid_api():
    col = request.args.get('collection', '').strip()
    alias = request.args.get('instance', '').strip()

    if not col or not VALID_NAME.match(col):
        return jsonify(error="Invalid collection name"), 400
    if not alias:
        return jsonify(error="Missing instance alias"), 400

    logger.warning(f"[Centroid API] Requested: collection={col!r}, instance={alias!r}")

    # Verify collection exists
    client = PipelineConfig.get_client()
    if not client or not client.is_live() or not client.is_ready():
        logger.error("Weaviate client not ready during /api/centroid")
        return jsonify(error="Weaviate not ready"), 503

    try:
        existing = client.collections.list_all(simple=True)
        if col not in existing:
            return jsonify(error="Collection not found"), 404
    except Exception as e:
        logger.error(f"Failed listing collections: {e}", exc_info=True)
        return jsonify(error="Error querying Weaviate collections"), 500

    cm = CentroidManager(
        instance_alias=alias,
        collection_name=col,
        base_path=cfg.paths.CENTROID_DIR
    )

    if request.method == 'POST':
        # Recalculate centroid
        try:
            vectors = cm.get_all_vectors(client, col)
            if not vectors:
                return jsonify(
                    ok=False,
                    error="No vectors found to calculate centroid.",
                    shape=None,
                    path=os.path.basename(cm.centroid_path)
                ), 200

            new_centroid = np.mean(np.vstack(vectors), axis=0)
            cm.save_centroid(new_centroid)

            return jsonify(
                ok=True,
                shape=list(cm.centroid.shape),
                path=os.path.basename(cm.centroid_path)
            ), 200

        except Exception as e:
            logger.error(f"Centroid calculation failed: {e}", exc_info=True)
            return jsonify(
                ok=False,
                error="Centroid calculation error",
                shape=None,
                path=os.path.basename(cm.centroid_path)
            ), 500

    # GET – return shape if exists
    cent = cm.get_centroid()
    if cent is None:
        return jsonify(
            loaded=False,
            shape=None,
            path=os.path.basename(cm.centroid_path)
        ), 200

    return jsonify(
        loaded=True,
        shape=list(cent.shape),
        path=os.path.basename(cm.centroid_path)
    ), 200


@app.route('/create_centroid', methods=['POST'])
def create_centroid():
    centroid_path = request.form.get('centroid_path')
    try:
        client = PipelineConfig.get_client()
        if not client or not client.is_ready():
            raise ConnectionError("Weaviate client not ready.")

        calculate_and_save_centroid(
            client,
            cfg.retrieval.COLLECTION_NAME,
            cfg.retrieval.WEAVIATE_ALIAS,
            os.path.dirname(centroid_path),
            force=True
        )
    finally:
        try: client.close()
        except: pass

    if os.path.exists(centroid_path):
        return jsonify({
            "status": "created",
            "message": f"Centroid saved to {centroid_path}"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "No vectors found in collection; centroid not created."
        }), 400


    


@app.route('/update_auto_domain_keywords', methods=['POST'])
def update_auto_domain_keywords():
    try:
        data = request.get_json(force=True)
        keywords = data.get('keywords', [])
        target_field = data.get('target_field', 'AUTO_DOMAIN_KEYWORDS')

        if not isinstance(keywords, list):
            return jsonify(success=False, error="Invalid 'keywords' format. Must be a list."), 400

        # Normalize + dedup
        keywords = sorted(set(map(str.strip, keywords)))

        if not hasattr(cfg, "env"):
            return jsonify(success=False, error="Config is missing 'env' section"), 500

        selected_n_top = getattr(cfg.env, "SELECTED_N_TOP", None)
        current = sorted(getattr(cfg.env, target_field, []) or [])

        # === DIRTY CHECK: No change ===
        if current == keywords:
            logger.info(f"[Config] No change in {target_field}. Skipping save.")
            return jsonify(success=True, changed=False)

        # === Apply update ===
        if target_field == "AUTO_DOMAIN_KEYWORDS":
            cfg.env.AUTO_DOMAIN_KEYWORDS = keywords
        elif target_field == "DOMAIN_KEYWORDS":
            cfg.env.DOMAIN_KEYWORDS = keywords
        else:
            return jsonify(success=False, error=f"Invalid target field: {target_field}"), 400

        if selected_n_top is not None:
            cfg.env.SELECTED_N_TOP = selected_n_top

        # Write to text file
        with open("auto_domain_keywords.txt", "w", encoding="utf-8") as f:
            f.write(", ".join(keywords))

        # Save to config YAML
        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))
        config_dict = dump_method()
        save_yaml_config(config_dict, CONFIG_YAML_PATH)

        logger.info(f"[Config] {target_field} updated with {len(keywords)} keywords.")
        return jsonify(success=True, changed=True)

    except Exception as e:
        logger.error(f"[Config] Failed to update {target_field}: {e}", exc_info=True)
        return jsonify(success=False, error=str(e)), 500


    

@app.route('/update_topn_config', methods=['POST'])
def update_topn_config():
    """Updates the config with the selected TopN value."""
    if cfg is None:
        return jsonify({"error": "Config unavailable."}), 503

    try:
        data = request.get_json()
        top_n = data.get('topN')

        if top_n is None:
            return jsonify({"success": False, "error": "No topN value provided."}), 400
        if not isinstance(top_n, int) or top_n < 1:
            return jsonify({"success": False, "error": "Invalid topN value."}), 400

        updates = {
            'env': {
                'SELECTED_N_TOP': top_n
            }
        }

        config_changed = cfg.update_and_save(updates)
        current_app.logger.info(f"TopN updated to {top_n}, config changed: {config_changed}")

        return jsonify({
            "success": True,
            "message": f"Updated configuration with TopN = {top_n}",
            "config_changed": config_changed
        })

    except Exception as e:
        current_app.logger.error(f"Error updating config with TopN: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-doc-count', methods=['GET'])
def get_doc_count():
    try:
        if cfg is None:
            return jsonify({"error": "Config unavailable."}), 503

        from weaviate.client import WeaviateClient
        from weaviate.connect import ConnectionParams
        from weaviate.config import AdditionalConfig, Timeout

        collection_name = cfg.retrieval.COLLECTION_NAME
        host            = cfg.retrieval.WEAVIATE_HOST
        http_port       = cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port       = cfg.retrieval.WEAVIATE_GRPC_PORT

        # Proper connection params: DO NOT omit or rename
        connection = ConnectionParams.from_params(
            http_host=host,
            http_port=http_port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False
        )

        client = WeaviateClient(
            connection_params=connection,
            additional_config=AdditionalConfig(timeout=Timeout(init=5, call=30))
        )
        client.connect()

        count = 0
        if client.collections.exists(collection_name):
            agg = client.collections.get(collection_name).aggregate
            count = agg.over_all(total_count=True).total_count

        client.close()
        return jsonify({"total_docs": count})

    except (weaviate.exceptions.WeaviateStartUpError,
            weaviate.exceptions.WeaviateConnectionError) as e:
        logger.error("Weaviate unavailable when fetching document count", exc_info=True)
        return jsonify({
            "total_docs": 0,
            "error": "Weaviate unavailable"
        }), 200
    except Exception as e:
        logger.exception("Unexpected error getting document count")
        return jsonify({
            "total_docs": 0,
            "error": str(e)
        }), 200





@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    # Clear chat history from session if you're storing it there
    if 'chat_history' in session:
        session['chat_history'] = []
        
    # Return success response
    return jsonify({"success": True})

@app.route("/inspect_instance", methods=["GET"])
def inspect_instance():
    client = None
    try:
        host       = cfg.retrieval.WEAVIATE_HOST
        http_port  = cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port  = cfg.retrieval.WEAVIATE_GRPC_PORT
        conn_to, call_to = cfg.retrieval.WEAVIATE_TIMEOUT

        params = ConnectionParams.from_params(
            http_host=host,
            http_port=http_port,
            grpc_host=host,
            grpc_port=grpc_port,
            http_secure=False,
            grpc_secure=False
        )

        client = WeaviateClient.connect(
            connection_params=params,
            additional_config=AdditionalConfig(timeout=Timeout(init=conn_to, call=call_to))
        )

        meta_info = client.connection.get("/schema").json()
        version = meta_info.get('version') or 'unknown'
        schema = meta_info.get("classes", [])

        classes_info = []
        for cls in schema:
            name  = cls.get("class")
            props = [{"name": p["name"], "dataType": p["dataType"]} for p in cls.get("properties", [])]
            vect  = cls.get("vectorizer")
            vect_cfg = cls.get("vectorizerConfig", {})

            try:
                coll  = client.collections.get(name)
                count = sum(1 for _ in coll.iterator(include_vector=False, return_properties=[]))
            except:
                count = None

            info = {
                "className":        name,
                "count":            count,
                "properties":       props,
                "vectorizer":       vect,
                "vectorizerConfig": vect_cfg
            }

            if any(p["name"] == "source" for p in props) and count:
                doc_counts = {}
                for obj in coll.iterator(return_properties=["source"]):
                    src_val = obj.properties.get("source")
                    if src_val is not None:
                        doc_counts[src_val] = doc_counts.get(src_val, 0) + 1

                distinct    = len(doc_counts)
                avg_chunks  = round(sum(doc_counts.values()) / distinct, 2) if distinct else 0
                info.update({
                    "distinctDocuments":    distinct,
                    "avgChunksPerDocument": avg_chunks
                })

            classes_info.append(info)

        client.close()
        return jsonify({"version": version, "classes": classes_info}), 200

    except Exception as e:
        if client:
            try: client.close()
            except: pass
        app.logger.error("Failed to inspect Weaviate instance", exc_info=True)
        return jsonify({"error": str(e)}), 500




# === Main Execution Block ===
if __name__ == '__main__':
    # Optional: diagnostic mode for CLI use (e.g., python app.py --diagnose)
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        print("=== Running standalone Weaviate inspection ===")
        with app.app_context():
            result, status = inspect_instance()
            print(f"\nStatus: {status}")
            import pprint
            pprint.pprint(result.json if hasattr(result, "json") else result)
        sys.exit(0)

    logger.info("Application starting...")

    # Ensure configuration loaded successfully
    if cfg is None:
        logger.critical("CRITICAL: Config failed load.")
        sys.exit(1)

    # --- Create important directories before bootstrapping anything heavy ---
    try:
        app.instance_path_obj = Path(app.instance_path)
        app.instance_path_obj.mkdir(parents=True, exist_ok=True)

        Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
        WEAVIATE_DATA_DIR_HOST.mkdir(parents=True, exist_ok=True)

        Path(cfg.paths.DOCUMENT_DIR).resolve().mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.critical(f"CRITICAL: Directory creation failed: {e}", exc_info=True)
        sys.exit(1)

    # ---- Application Context Starts Here ----
    with app.app_context():
        logger.info("Entering application context for initialization.")

        # --- Initialize Flask extensions exactly once ---
        try:
            if not hasattr(db, 'app') or db.app is None:
                db.init_app(app)
            logger.info("Flask extensions initialized.")
        except Exception as ext_e:
            logger.critical(f"Failed extensions init: {ext_e}", exc_info=True)
            sys.exit(1)

        # --- Setup database tables ---
        logger.info("Checking/creating database tables...")
        try:
            db.create_all()
            logger.info("DB tables checked/created.")
        except Exception as db_e:
            logger.error(f"Failed to create DB tables: {db_e}", exc_info=True)

        initialize_pipeline(app.app_context())
        load_presets()
        logger.info(f"Presets loaded in main block. Count: {len(presets)}")

    # --- Define cleanup logic to release Weaviate gRPC resources on shutdown ---
    import atexit

    def cleanup_resources():
        # Close the retriever on the live pipeline instance
        try:
            from pipeline import PIPELINE_INSTANCE
            inst = PIPELINE_INSTANCE
        except ImportError:
            inst = None

        if inst and hasattr(inst.retriever, 'close'):
            try:
                inst.retriever.close()
                logger.info("Closed pipeline retriever client")
            except Exception as e:
                logger.error(f"Error closing retriever: {e}")

        # Also close any residual Weaviate clients
        try:
            import weaviate
            for attr in dir(weaviate):
                if attr.endswith("Client"):
                    cls = getattr(weaviate, attr, None)
                    for client in getattr(cls, "_instances", []):
                        try:
                            client.close()
                            logger.info(f"Closed {attr} instance")
                        except Exception as e:
                            logger.error(f"Error closing {attr} instance: {e}")
        except ImportError:
            logger.warning("Weaviate module not found during cleanup.")
        except Exception as e:
            logger.error(f"Error during additional Weaviate cleanup: {e}", exc_info=True)

    atexit.register(cleanup_resources)

    # --- Final status logging before run ---
    try:
        from pipeline import PIPELINE_INSTANCE
        pipeline_ready = True
    except ImportError:
        pipeline_ready = False

    if not pipeline_ready:
        logger.warning("Pipeline initialization failed or skipped.")
    if not docker_available:
        logger.warning("Docker client unavailable.")

    logger.info(f"Flask-Session: Type={app.config.get('SESSION_TYPE')}, Dir={app.config.get('SESSION_FILE_DIR')}")
    logger.info(f"Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI')}")

    # --- Launch Flask development server ---
    logger.info(f"Starting Flask server on http://0.0.0.0:5000/ (Debug: {app.debug})")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
