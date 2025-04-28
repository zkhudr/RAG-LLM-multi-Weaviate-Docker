import os
# Disable optional telemetry
os.environ['POSTHOG_DISABLED'] = 'true'

import logging
import json
import traceback
import docker
from docker.errors import NotFound as DockerNotFound, APIError as DockerAPIError
from werkzeug.utils import secure_filename
import yaml
import csv
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_session import Session # Import Session from flask_session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import re
import shutil
import weaviate
from weaviate.exceptions import WeaviateConnectionError # Use for specific catching
import urllib.parse
from logging.handlers import RotatingFileHandler # Use rotating handler for safety
import sys # For StreamHandler


load_dotenv()


# --- Configure Flask App ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key_for_dev_only')
if app.secret_key == 'fallback_secret_key_for_dev_only':
    logger.warning("Using fallback Flask secret key. Set FLASK_SECRET_KEY environment variable for production.")
# --- End Flask App Config ---

log_level = logging.INFO # Or DEBUG based on needs
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Use instance path for log file - NOW 'app' exists
log_file_path = os.path.join(app.instance_path, 'app.log')
# Ensure instance path exists BEFORE creating handler
try:
    os.makedirs(app.instance_path, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Failed to create instance path '{app.instance_path}': {e}", file=sys.stderr)
  
# File Handler (Rotating)
# Rotate logs at 5MB, keep 3 backups
file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(log_level)

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(log_level)

# Get the root logger and configure it
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.handlers.clear() # Remove any default handlers first
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Optional: Quieten noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("weaviate").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("pdfplumber").setLevel(logging.WARNING) # Uncomment if needed

logger = logging.getLogger(__name__) # Get logger for app.py itself
logger.info("Logging configured: Level=%s, File=%s", logging.getLevelName(log_level), log_file_path)
# --- END Logging Configuration Block ---

# --- Configure Database ---
db = SQLAlchemy()
db_path = os.path.join(app.instance_path, 'chat_history.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Configure Flask-Session ---
sess = Session()
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_DIR"] = "./.flask_session"
app.config["SESSION_FILE_THRESHOLD"] = 100

# --- Multi-Instance Constants ---
WEAVIATE_STATE_FILE = Path("./weaviate_state.json").resolve()
DEFAULT_BASE_HTTP_PORT = 8090
DEFAULT_BASE_GRPC_PORT = 51001
WEAVIATE_IMAGE = "semitechnologies/weaviate:1.25.1"
WEAVIATE_DATA_DIR_HOST = Path("./weaviate_data").resolve()

# --- Docker Client Initialization ---
docker_available = False
docker_client = None
try:
    docker_client = docker.from_env()
    docker_client.ping()
    docker_available = True
    logger.info("Docker client initialized and connected successfully.")
except Exception as docker_err:
    logger.error(f"Failed to initialize Docker client: {docker_err}. Multi-instance creation disabled.")

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
config_available = False
cfg = None
AppConfig = None
FlowStyleList = list
try:
    from config import cfg, AppConfig, FlowStyleList
    config_available = True
    logger.info("Successfully imported configuration (cfg, AppConfig).")
except ImportError:
    logger.error("CRITICAL: Failed to import configuration (cfg, AppConfig, FlowStyleList). Check config.py.", exc_info=True)
    exit(1)

pipeline_available = False
pipeline = None
IndustrialAutomationPipeline = None
try:
    from pipeline import IndustrialAutomationPipeline
    pipeline_available = True
    logger.info("Successfully imported IndustrialAutomationPipeline.")
except ImportError:
    logger.error("Failed to import IndustrialAutomationPipeline. Check pipeline.py.", exc_info=True)


# --- CORRECTED Ingestion Script Imports ---
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
    from ingest_block import run_ingestion as run_incremental_ingestion_func # Use alias
    run_incremental_ingestion = run_incremental_ingestion_func # Assign to the expected variable name
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

# --- Pipeline Initialization Function ---
def initialize_pipeline(app_context=None):
    """Initializes or re-initializes the global pipeline object."""
    global pipeline, pipeline_available
    context = app_context if app_context else app.app_context()
    with context:
        logger.info("Attempting to initialize/re-initialize pipeline...")
        if not pipeline_available: logger.error("Skip init: Pipeline module unavailable."); pipeline = None; return
        if not config_available or not cfg: logger.error("Skip init: Config unavailable."); pipeline = None; return
        try:
            # ... (Close old client if exists) ...
            pipeline = IndustrialAutomationPipeline() # Instantiate
            logger.info("Pipeline initialized/re-initialized successfully.")
            # ... (Log connection status checks) ...
        except Exception as e:
            pipeline = None
            logger.error(f"Failed to initialize/re-initialize Pipeline: {e}", exc_info=True)

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
    

def save_presets(presets_data, filename=PRESETS_FILE):
    """Saves presets dict to JSON file."""
    global presets # Optional: update global after save too
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, indent=2)
        presets = presets_data # Update global variable after successful save
        logger.info(f"Presets saved to '{filename}'.")
        return True # Indicate success
    except IOError as e:
        logger.error(f"Failed to save presets to '{filename}': {e}")
        return False # Indicate failure)

# === NEW: Multi-Instance Helper Functions ===

def save_config_to_yaml(config_dict: Dict[str, Any]) -> bool:
        """Validates and saves the config dictionary to the YAML file."""
        global cfg # Ensure we update the global cfg instance
        if not config_available or not AppConfig:
            logging.error("Cannot save YAML: Config system (cfg or AppConfig) not available.")
            raise RuntimeError("Configuration system not loaded.")

        try:
            # 1. Validate the incoming dictionary by creating a new AppConfig instance
            validated_config = AppConfig(**config_dict)

            # 2. Update the global 'cfg' instance
            # This is slightly tricky as Pydantic models are often immutable after creation.
            # Easiest might be to reassign 'cfg' if AppConfig loading allows it,
            # or update section by section if 'cfg' must remain the same object.
            # Let's try section-by-section update:
            for section_name in validated_config.model_fields.keys():
                setattr(cfg, section_name, getattr(validated_config, section_name))

            # 3. Prepare the dictionary for YAML dump (using validated data)
            # Use model_dump() for Pydantic v2, or .dict() for v1
            dump_dict = validated_config.model_dump() if hasattr(validated_config, 'model_dump') else validated_config.dict()

            # Handle FlowStyleList for keywords before dumping
            if FlowStyleList and 'env' in dump_dict:
                for key in ['DOMAIN_KEYWORDS', 'AUTO_DOMAIN_KEYWORDS', 'USER_ADDED_KEYWORDS']:
                    if key in dump_dict['env'] and isinstance(dump_dict['env'][key], list):
                        dump_dict['env'][key] = FlowStyleList(dump_dict['env'][key])


            # 4. Atomic write to YAML using standard Dumper
            temp_path = f"{CONFIG_YAML_PATH}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                # Use default_flow_style=False for block style unless FlowStyleList is used
                yaml.dump(dump_dict, f, indent=2, sort_keys=False, default_flow_style=None)
            os.replace(temp_path, CONFIG_YAML_PATH)
            logging.info(f"Configuration saved successfully to {CONFIG_YAML_PATH}")
            return True

        except ValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise # Re-raise for Flask route to catch
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
        if not cfg: return None
        current_host = cfg.retrieval.WEAVIATE_HOST
        current_http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        logger = app.logger # Use app logger

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
@app.route("/", methods=["GET"])
def index(): return render_template("index.html")

@app.route('/get_config', methods=['GET'])
def get_config_api():
    if not config_available or not cfg: 
        return jsonify({"error": "Config unavailable."}), 503
    try:
        # Initialize the config_dict in case of an error
        config_dict = {}

        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))
        if not dump_method: 
            raise TypeError("Config object cannot be serialized.")
        
        # Serialize the config, excluding sensitive data
        config_dict = dump_method(exclude={'security': {'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'}})

        # Add default pipeline section if missing
        if 'pipeline' not in config_dict:
            logger.warning("Pipeline section missing in config. Adding default values.")
            config_dict['pipeline'] = {'max_history_turns': 5}  # Default pipeline values

        # Debug: Log the config data before sending it
        logger.info(f"Config to be returned: {config_dict}")
        
        return jsonify(config_dict)
    except Exception as e: 
        logger.error(f"API /get_config error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route('/list_presets', methods=['GET'])
def list_presets_api():
    global presets
    return jsonify(presets)

@app.route('/delete_preset/<path:preset_name>', methods=['DELETE'])
def delete_preset_api(preset_name):
    """API Endpoint: Deletes a specified preset."""
    global presets # Still need global access to potentially update it
    logger = app.logger

    # Decode the preset name
    try:
        decoded_preset_name = urllib.parse.unquote(preset_name)
    except Exception as decode_err:
        logger.error(f"API /delete_preset: Error decoding preset name '{preset_name}': {decode_err}")
        # Send JSON 400 (ensure browser doesn't get HTML here)
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
    """API Endpoint: Returns the status of configured API keys."""
    logger = app.logger # Use app's logger
    if not config_available or not cfg or not cfg.security:
        logger.warning("API /api/key_status: Config/Security settings unavailable.")
        # Return a consistent structure even on error, or a 503
        return jsonify({
            "deepseek": False,
            "openai": False,
            "anthropic": False,
            "cohere": False,
            "error": "Configuration unavailable"
        }), 503 # Service Unavailable might be appropriate

    try:
        status = {
            # Check if the key exists and has a non-empty string value
            "deepseek": bool(getattr(cfg.security, 'DEEPSEEK_API_KEY', None)),
            "openai": bool(getattr(cfg.security, 'OPENAI_API_KEY', None)),
            "anthropic": bool(getattr(cfg.security, 'ANTHROPIC_API_KEY', None)),
            "cohere": bool(getattr(cfg.security, 'COHERE_API_KEY', None))
            # Add other keys here if needed
        }
        logger.info(f"API /api/key_status: Returning status: {status}")
        return jsonify(status), 200

    except Exception as e:
        logger.error(f"API /api/key_status: Unexpected error checking key status: {e}", exc_info=True)
        # Return a consistent error structure
        return jsonify({
            "deepseek": False,
            "openai": False,
            "anthropic": False,
            "cohere": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


# --- Action Endpoints (JSON based) ---
@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    if not pipeline_available or not pipeline: return jsonify({"role": "assistant", "text": "Error: Pipeline unavailable.", "error": True}), 503
    try:
        data = request.get_json(); assert data and data.get("query")
        user_query = data["query"].strip(); assert user_query
        chat_history = session.get('chat_history', [])
        result_dict = pipeline.generate_response(query=user_query, chat_history=chat_history)
        assistant_message = {"role": "assistant", "text": result_dict.get("response", "..."), "sources": result_dict.get("sources"), "error": result_dict.get("error", False), "timestamp": datetime.now().isoformat()}
        chat_history.append(assistant_message)
        session['chat_history'] = chat_history
        return jsonify(assistant_message)
    except Exception as e: logger.error(f"Error in /run_pipeline: {str(e)}", exc_info=True); return jsonify({"role": "assistant", "text": f"Internal error: {str(e)}", "error": True}), 500

@app.route("/apply_preset/<preset_name>", methods=["POST"])
def apply_preset(preset_name):
    global presets
    if preset_name not in presets: return jsonify({"success": False, "error": "Preset not found"}), 404
    if not config_available or not cfg: return jsonify({"success": False, "error": "Config system unavailable."}), 503
    try:
        preset_data = presets[preset_name]
        config_changed = cfg.update_and_save(preset_data)
        if config_changed:
            initialize_pipeline(app.app_context())
            if not pipeline: logger.error("Pipeline failed re-init after preset!") # Log error if re-init fails
        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))
        current_config_dict = dump_method(exclude={'security': {'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'}}) if dump_method else {}
        return jsonify({"success": True, "message": f"Preset '{preset_name}' applied.", "config": current_config_dict})
    except ValidationError as e: logger.error(f"Preset validation error '{preset_name}': {e}"); details = "..."; return jsonify({"success": False, "error": f"Invalid preset: {details}"}), 400
    except Exception as e: logger.error(f"Preset apply error '{preset_name}': {e}", exc_info=True); return jsonify({"success": False, "error": str(e)}), 500

@app.route("/save_preset", methods=["POST"])
def save_preset_api():
    global presets
    if not config_available: return jsonify({"success": False, "error": "Config unavailable."}), 503
    try:
        data = request.get_json(); assert data
        preset_name = data.get("preset_name", "").strip(); assert preset_name
        preset_data = data.get("config"); assert preset_data and isinstance(preset_data, dict)
        _ = AppConfig(**preset_data) # Validate
       ################################################################################################################### #presets = load_presets() # Reload before update
        presets[preset_name] = preset_data
        if save_presets(presets): return jsonify({"success": True, "message": f"Preset '{preset_name}' saved."})
        else: return jsonify({"success": False, "error": "Failed to write presets file."}), 500
    except ValidationError as ve_preset: logger.error(f"Preset data validation failed: {ve_preset}"); details = "..."; return jsonify({"success": False, "error": f"Invalid preset data: {details}"}), 400
    except Exception as e: logger.error(f"Preset save API error: {str(e)}", exc_info=True); return jsonify({"success": False, "error": str(e)}), 500

@app.route('/save_config', methods=['POST'])
def save_config_api():
    """API Endpoint: Saves configuration received as JSON."""
    logger = app.logger # Use app's logger
    if not config_available or not cfg:
        logger.error("API /save_config: Config system unavailable.")
        return jsonify({"success": False, "error": "Configuration system not available."}), 503

    try:
        updates_dict = request.get_json() # Get JSON data from request body
        if not updates_dict:
            logger.warning("API /save_config: No JSON data received.")
            return jsonify({"success": False, "error": "No JSON data received."}), 400

        logger.info("API /save_config: Received config data via JSON. Attempting update and save...")
        # Use update_and_save which includes validation within config.py
        config_changed = cfg.update_and_save(updates_dict)

        if config_changed:
            logger.info("API /save_config: Configuration updated and saved successfully.")
            # Decide if pipeline needs re-init based on changes (optional, depends on config impact)
            initialize_pipeline(app.app_context()) # Example: Re-init if needed
            if pipeline and pipeline.retriever: # Basic check if pipeline object exists
                logger.info("API /save_config: Pipeline re-initialized successfully.")
            else:
                 logger.error("API /save_config: Pipeline failed to re-initialize after config change!")
            return jsonify({"success": True, "message": "Configuration saved successfully!"})
        else:
            logger.info("API /save_config: No configuration changes detected from received data.")
        return jsonify({"success": True, "message": "No configuration changes detected."})

    except ValidationError as e: # Catch Pydantic validation errors
        logger.error(f"API /save_config: Validation Error saving config: {e}", exc_info=False) # Log concisely
        # Format error details for user feedback
        error_details = ". ".join([f"{'.'.join(map(str, err.get('loc', ['?'])))}: {err.get('msg', 'invalid')}" for err in e.errors()])
        return jsonify({"success": False, "error": f"Invalid configuration values: {error_details}"}), 400
    except Exception as e: # Catch other unexpected errors
        logger.error(f"API /save_config: Unexpected error saving configuration: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Failed to save configuration: {str(e)}"}), 500

@app.route('/upload_files', methods=['POST'])
def upload_files():
    # ... (Keep implementation using request.files) ...
    if not config_available: return jsonify({"success": False, "error": "Config unavailable"}), 500
    try: upload_dir = Path(cfg.paths.DOCUMENT_DIR).resolve(); upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as dir_e: return jsonify({"success": False, "error": f"Dir error: {dir_e}"}), 500
    try:
        files = request.files.getlist("files"); assert files and not all(not f or f.filename == '' for f in files)
        saved_files = []
        for file in files:
            if file and file.filename: filename = secure_filename(file.filename); file.save(upload_dir / filename); saved_files.append(filename)
        return jsonify({"success": True, "files": saved_files})
    except Exception as e: return jsonify({"success": False, "error": f"Upload failed: {e}"}), 500

# --- CORRECTED Ingestion Routes ---

@app.route('/start_ingestion', methods=['POST'])
def start_ingestion():
    """(Corrected) Triggers FULL document ingestion using available components."""
    logger = app.logger
    # Check if the necessary components were imported
    if not ingest_full_available or not DocumentProcessor or not PipelineConfig:
        logger.error("/start_ingestion: Required ingestion components missing due to import errors (or failed import).") 
        return jsonify({'success': False, 'error': 'Full ingestion components unavailable (Import Error)'}), 503
    if not config_available or not cfg:
        return jsonify({'success': False, 'error': 'Configuration system not available.'}), 500

    weaviate_client = None
    try:
        logger.info(f"Full Ingest: Getting client via PipelineConfig for {cfg.retrieval.WEAVIATE_HOST}:{cfg.retrieval.WEAVIATE_HTTP_PORT}")
        # --- Use PipelineConfig.get_client ---
        weaviate_client = PipelineConfig.get_client()

        # Check connection (is_ready/is_live)
        if not weaviate_client.is_ready(): # Adapt if needed for client version
            raise WeaviateConnectionError(f"Failed connect Weaviate at {cfg.retrieval.WEAVIATE_HOST}:{cfg.retrieval.WEAVIATE_HTTP_PORT}")
        logger.info("Weaviate connected for full ingestion.")

        doc_dir = Path(cfg.paths.DOCUMENT_DIR).resolve()
        if not doc_dir.is_dir(): raise FileNotFoundError(f"Ingestion dir not found: {doc_dir}")

        logger.info(f"Starting full ingestion process for directory: {doc_dir}")
        processor = DocumentProcessor(data_dir=str(doc_dir), client=weaviate_client)
        processor.execute()

        stats = getattr(processor, 'get_stats', lambda: {"message": "Stats unavailable"})()
        logger.info(f"Full ingestion process finished. Stats: {stats}")
        return jsonify({'success': True, 'message': 'Full document ingestion successful', 'stats': stats})

    except FileNotFoundError as fnf_e: logger.error(f"Ingestion failed: {fnf_e}"); return jsonify({'success': False, 'error': str(fnf_e)}), 404
    except (WeaviateConnectionError, ConnectionError) as conn_e: logger.error(f"Ingestion Weaviate conn failed: {conn_e}"); return jsonify({'success': False, 'error': str(conn_e)}), 503
    except NameError as ne: logger.error(f"Ingestion failed due to missing component: {ne}"); return jsonify({'success': False, 'error': f'Missing Component: {ne}'}), 503
    except Exception as e: logger.error(f"Ingestion failed: {str(e)}", exc_info=True); return jsonify({'success': False, 'error': f"Ingestion process failed: {str(e)}"}), 500
    finally:
        # Close client connection
        if weaviate_client and hasattr(weaviate_client, 'close') and callable(weaviate_client.close):
            try: weaviate_client.close(); logger.info("Weaviate client closed after full ingestion.")
            except Exception as close_e: logger.error(f"Error closing Weaviate client: {close_e}")
        elif weaviate_client: logger.info("Weaviate client has no close method.")


@app.route("/ingest_block", methods=["POST"])
def ingest_block_route():
    """(Corrected) Triggers INCREMENTAL ingestion using available components."""
    logger = app.logger
    if not ingest_block_available or not run_incremental_ingestion:
        logger.error("/ingest_block: Required 'run_incremental_ingestion' component missing")
        return jsonify({"success": False, "error": "Incremental ingestion components unavailable"}), 503
    if not config_available or not cfg:
        return jsonify({"success": False, "error": "Configuration system not available."}), 500

    try:
        # Get document directory from config
        document_dir = cfg.paths.DOCUMENT_DIR
        logger.info(f"Starting incremental ingestion for folder: {document_dir}")
        
        # Call run_ingestion with the folder path
        result = run_incremental_ingestion(document_dir)
        
        logger.info(f"Incremental ingestion finished. Result: {result}")
        return jsonify({
            "success": True,
            "message": result.get("message", "Incremental ingestion completed"),
            "stats": {
                "processed": result.get("processed", 0),
                "errors": 0
            }
        })
    except Exception as e:
        logger.error(f"Error during incremental ingestion: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "stats": {
                "processed": 0,
                "errors": 1
            }
        }), 500

# --- Chat History Routes ---
# UPDATED: Use JSON for save/delete actions

@app.route('/run_keyword_builder', methods=['POST'])
def run_keyword_builder():
    """Runs the keyword_builder_v3.py script with provided parameters."""
    logger = app.logger
    
    # Check for Docker availability
    if not docker_available:
        return jsonify({"success": False, "error": "Docker client unavailable."}), 503
    
    # Check for configuration availability
    if not config_available or not cfg:
        return jsonify({"success": False, "error": "Config system unavailable."}), 503
    
    try:
        data = request.get_json()
        
        # Use config defaults if not specified in request
        dke_cfg = getattr(cfg, "domain_keyword_extraction", None)
        keybert_model = data.get('keybert_model') or (dke_cfg.keybert_model if dke_cfg else 'all-MiniLM-L6-v2')
        top_n_per_doc = int(data.get('top_n_per_doc') or (dke_cfg.top_n_per_doc if dke_cfg else 10))
        final_top_n = int(data.get('final_top_n') or (dke_cfg.final_top_n if dke_cfg else 100))
        min_doc_freq = int(data.get('min_doc_freq') or (dke_cfg.min_doc_freq if dke_cfg else 2))
        diversity = float(data.get('diversity') or (dke_cfg.diversity if dke_cfg else 0.7))
        no_pos_filter = data.get('no_pos_filter') if 'no_pos_filter' in data else (dke_cfg.no_pos_filter if dke_cfg else False)

        # Get parameters with defaults
        keybert_model = data.get('keybert_model', 'all-MiniLM-L6-v2')
        top_n_per_doc = int(data.get('top_n_per_doc', 10))
        final_top_n = int(data.get('final_top_n', 100))
        min_doc_freq = int(data.get('min_doc_freq', 2))
        diversity = float(data.get('diversity', 0.7))
        no_pos_filter = data.get('no_pos_filter', False)
        
        # Get current Weaviate connection details from cfg
        collection_name = cfg.retrieval.COLLECTION_NAME
        weaviate_http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        
        # Check Weaviate connection before proceeding
        if not check_weaviate_connection(weaviate_http_port):
            return jsonify({
                "success": False, 
                "error": f"Cannot connect to Weaviate on port {weaviate_http_port}. Please ensure the service is running."
            }), 503
        
        # Build command with parameters
        cmd = [
            sys.executable,
            "keywords_builder_v3.py",
            "--collection", collection_name,
            "--keybert_model", keybert_model,
            "--top_n_per_doc", str(top_n_per_doc),
            "--final_top_n", str(final_top_n),
            "--min_doc_freq", str(min_doc_freq),
            "--diversity", str(diversity)
        ]
        
        if no_pos_filter:
            cmd.append("--no_pos_filter")
        
        # Add any additional parameters that might be in the request
        for param, value in data.items():
            if param not in ['keybert_model', 'top_n_per_doc', 'final_top_n', 'min_doc_freq', 'diversity', 'no_pos_filter']:
                if value is not None:
                    cmd.extend([f"--{param}", str(value)])
        
        # Execute the script
        import subprocess
        import tempfile
        import time
        import signal
        from threading import Timer
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        logger.info(f"Running keyword builder with command: {' '.join(cmd)}")
        
        # Get timeout from config or use default
        timeout_seconds = getattr(cfg.security, 'APITIMEOUT', 60)
        
        # Start the process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ.copy()  # Ensure environment variables are passed
        )
        
        # Set up timeout handling
        process_timed_out = [False]
        
        def kill_process():
            process_timed_out[0] = True
            process.kill()
        
        timer = Timer(timeout_seconds, kill_process)
        
        try:
            timer.start()
            stdout, stderr = process.communicate()
        finally:
            timer.cancel()
        
        # Check if process timed out
        if process_timed_out[0]:
            logger.error(f"Keyword builder process timed out after {timeout_seconds} seconds")
            return jsonify({
                "success": False, 
                "error": f"Process timed out after {timeout_seconds} seconds",
                "details": "The keyword extraction process took too long to complete."
            }), 504  # Gateway Timeout
        
        # Check if process failed
        if process.returncode != 0:
            logger.error(f"Keyword builder failed with error: {stderr}")
            
            # Attempt to extract more specific error information
            error_message = "Unknown error"
            if "No such file or directory" in stderr:
                error_message = "Script file not found or not accessible"
            elif "ModuleNotFoundError" in stderr:
                error_message = "Missing Python module. Please check dependencies."
            elif "ConnectionError" in stderr or "Connection refused" in stderr:
                error_message = "Connection error. Please check Weaviate is running."
            elif "MemoryError" in stderr:
                error_message = "Process ran out of memory. Try reducing batch size."
            else:
                # Extract the last few lines of stderr for more context
                error_lines = stderr.strip().split('\n')[-3:]
                error_message = '\n'.join(error_lines)
            
            return jsonify({
                "success": False, 
                "error": f"Process failed with exit code {process.returncode}",
                "details": error_message,
                "full_error": stderr
            }), 500
        
        # Parse the output to extract keywords and scores
        keywords = []
        in_keywords_section = False
        
        for line in stdout.splitlines():
            if "--- Top" in line and "Filtered Domain Keywords" in line:
                in_keywords_section = True
                continue
            elif "-----------------------------------------------------------------------------" in line:
                in_keywords_section = False
                continue
            
            if in_keywords_section:
                parts = line.strip().split(" : ")
                if len(parts) == 2:
                    term = parts[0].strip()
                    try:
                        score = float(parts[1].strip())
                        keywords.append({"term": term, "score": score})
                    except ValueError:
                        logger.warning(f"Failed to parse score for keyword: {line}")
                        continue
        
        # Check if we found any keywords
        if not keywords:
            logger.warning("No keywords were extracted from the output")
            
            # Check if there's any useful information in the output
            if stdout.strip():
                # Look for any lines that might contain useful information
                info_lines = [line for line in stdout.splitlines() 
                             if "error" in line.lower() or 
                                "warning" in line.lower() or 
                                "exception" in line.lower()]
                
                if info_lines:
                    return jsonify({
                        "success": False,
                        "error": "Failed to extract keywords",
                        "details": "\n".join(info_lines)
                    }), 500
            
            return jsonify({
                "success": False,
                "error": "No keywords were extracted",
                "details": "The script completed successfully but no keywords were found in the output."
            }), 500
        
        logger.info(f"Extracted {len(keywords)} keywords from keyword builder output")
        
        # Check if caching is enabled
        cache_enabled = getattr(cfg.security, 'CACHEENABLED', False)
        if cache_enabled:
            # Cache the results with a timestamp
            cache_key = f"keywords_{collection_name}_{int(time.time())}"
            try:
                cache_results(cache_key, {
                    "keywords": keywords,
                    "parameters": {
                        "keybert_model": keybert_model,
                        "top_n_per_doc": top_n_per_doc,
                        "final_top_n": final_top_n,
                        "min_doc_freq": min_doc_freq,
                        "diversity": diversity,
                        "no_pos_filter": no_pos_filter
                    },
                    "timestamp": time.time()
                })
                logger.info(f"Cached keyword results with key: {cache_key}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache keyword results: {cache_error}")
        
        # Save the raw output for debugging purposes
        try:
            output_dir = os.path.join(os.path.dirname(__file__), "data", "outputs")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"keyword_builder_output_{timestamp}.txt")
            
            with open(output_file, 'w') as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n\n")
                f.write(f"STDOUT:\n{stdout}\n\n")
                f.write(f"STDERR:\n{stderr}\n")
            
            logger.info(f"Saved raw output to {output_file}")
        except Exception as save_error:
            logger.warning(f"Failed to save raw output: {save_error}")
        
        # Return successful response with extracted keywords
        return jsonify({
            "success": True,
            "keywords": keywords,
            "count": len(keywords),
            "parameters": {
                "keybert_model": keybert_model,
                "top_n_per_doc": top_n_per_doc,
                "final_top_n": final_top_n,
                "min_doc_freq": min_doc_freq,
                "diversity": diversity,
                "no_pos_filter": no_pos_filter
            },
            "message": f"Successfully extracted {len(keywords)} domain keywords"
        })
        
    except ValueError as ve:
        logger.error(f"Value error in keyword builder: {ve}", exc_info=True)
        return jsonify({"success": False, "error": f"Invalid parameter value: {str(ve)}"}), 400
    except Exception as e:
        logger.error(f"Error running keyword builder: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def check_weaviate_connection(port=8080):
    """Check if Weaviate is accessible on the specified port."""
    try:
        import requests
        from requests.exceptions import RequestException
        
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

# Add comprehensive cleanup code for all Weaviate clients
import atexit
def cleanup_resources():
        # Close pipeline retriever client
        if 'pipeline' in globals() and pipeline:
            if hasattr(pipeline.retriever, 'close'):
                try:
                    pipeline.retriever.close()
                    logger.info("Closed pipeline retriever client")
                except Exception as e:
                    logger.error(f"Error closing retriever: {e}")
        
        # Close any other Weaviate clients
        try:
            logger.info("Attempting to close any remaining Weaviate connections")
            import weaviate
            # Find all client instances that might be open
            for client_attr in dir(weaviate):
                if client_attr.endswith('Client'):
                    client_class = getattr(weaviate, client_attr, None)
                    if client_class and hasattr(client_class, '_instances'):
                        for client in list(getattr(client_class, '_instances', [])):
                            try:
                                if hasattr(client, 'close') and callable(client.close):
                                    client.close()
                                    logger.info(f"Closed {client_attr} instance")
                            except Exception as e:
                                logger.error(f"Error closing {client_attr} instance: {e}")
        except Exception as e:
            logger.error(f"Error during additional Weaviate connection cleanup: {e}")
atexit.register(cleanup_resources)
   
@app.route('/update_config_keywords', methods=['POST'])
def update_config_keywords():
    """Updates the config with extracted keywords."""
    logger = app.logger
    
    if not config_available or not cfg:
        return jsonify({"success": False, "error": "Config system unavailable."}), 503
    
    try:
        data = request.get_json()
        keywords = data.get('keywords', [])
        
        if not keywords:
            return jsonify({"success": False, "error": "No keywords provided"}), 400
        
        # Create updates dictionary for config
        updates = {
            'env': {
                'AUTO_DOMAIN_KEYWORDS': keywords
            }
        }
        
        # Update config
        config_changed = cfg.update_and_save(updates)
        
        if config_changed:
            # Re-initialize pipeline if config changed
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
    logger = app.logger
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


@app.route('/load_chat/<int:chat_id>', methods=['GET']) # Corrected route param type
def load_chat(chat_id):
    """Loads a specific chat history into the current session."""
    logger = app.logger
    try:
        chat = db.session.get(SavedChat, chat_id)
        if chat:
            session['chat_history'] = chat.history
            logger.info(f"Loaded chat '{chat.name}' (ID: {chat_id}) into session.")
            # Return the loaded data for frontend confirmation/use
            return jsonify({"success": True, "id": chat.id, "name": chat.name, "history": chat.history})
        else:
            return jsonify({"success": False, "error": "Chat not found."}), 404
    except Exception as e:
        logger.error(f"Error loading chat ID {chat_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500


@app.route('/list_chats', methods=['GET'])
def list_chats():
    """Lists all saved chats (ID, Name, Timestamp)."""
    logger = app.logger
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
    logger = app.logger
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


# --- Multi-Weaviate Routes ---
# UPDATED: Ensure JSON handling

@app.route("/create_weaviate_instance", methods=["POST"])
def create_weaviate_instance():
    logger = app.logger
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
    logger = app.logger
    instances_details = []
    
    try:
        # First, get default config details
        default_instance = None
        try:
            with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            default_host = yaml_config.get('retrieval', {}).get('WEAVIATE_HOST', 'localhost')
            try: default_http_port = int(yaml_config.get('retrieval', {}).get('WEAVIATE_HTTP_PORT', 8080))
            except: default_http_port = 8080
            try: default_grpc_port = int(yaml_config.get('retrieval', {}).get('WEAVIATE_GRPC_PORT', 50051))
            except: default_grpc_port = 50051
            
            # Check if current runtime config matches this default
            is_active = (cfg and cfg.retrieval.WEAVIATE_HOST == default_host and 
                        cfg.retrieval.WEAVIATE_HTTP_PORT == default_http_port)
            
            default_instance = {
                "name": "Active (from current config)", 
                "host": default_host, 
                "http_port": default_http_port,
                "grpc_port": default_grpc_port, 
                "status": "config_default", 
                "container_id": None, 
                "container_name": None,
                "active": is_active
            }
        except Exception as cfg_read_err: 
            logger.error(f"Could not read default config for listing: {cfg_read_err}")
        
        # Get all managed instances
        managed_instances = []
        instance_state = load_weaviate_state()
        for name, details in instance_state.items():
            container_status = details.get("status", "unknown")
            container_id = details.get("container_id")
            is_active = (cfg and cfg.retrieval.WEAVIATE_HOST == details.get('host') and 
                        cfg.retrieval.WEAVIATE_HTTP_PORT == details.get('http_port'))
            
            if docker_available and container_id:
                try:
                    container = docker_client.containers.get(container_id)
                    container_status = container.status
                except DockerNotFound: 
                    container_status = "not_found"
                except Exception as docker_err: 
                    logger.warning(f"Docker error checking status for {container_id}: {docker_err}")
                    container_status = "check_error"
            
            managed_instances.append({
                "name": name, 
                "host": details.get("host", "?"), 
                "http_port": details.get("http_port", "?"),
                "grpc_port": details.get("grpc_port", "?"), 
                "status": container_status, 
                "container_id": container_id,
                "container_name": details.get("container_name"),
                "active": is_active
            })
        
        # Combine default and managed instances
        if default_instance:
            # Find the active managed instance that matches the default config
            # Find the active managed instance that matches the default config
            active_managed_instance = None
            for instance in managed_instances:
                if (instance.get('host') == default_instance.get('host') and
                    instance.get('http_port') == default_instance.get('http_port')):
                    active_managed_instance = instance
                    break

            if active_managed_instance:
                # Replace default instance name with active managed instance name
                combined_instance = default_instance.copy()
                combined_instance['name'] = f"Active: instance - \"{active_managed_instance['name']}\""
                combined_instance['active'] = True
                
                # Remove the active managed instance from the list to avoid duplication
                managed_instances = [inst for inst in managed_instances if inst != active_managed_instance]
                
                # Add combined instance to the beginning of the list
                instances_details = [combined_instance] + managed_instances
            else:
                # No active managed instance matches default, just use all instances
                instances_details = [default_instance] + managed_instances
        else:
            instances_details = managed_instances
        
        # Sort instances (ensure default/active is first)
        return jsonify(sorted(instances_details, 
                             key=lambda x: (not x.get('active', False), x.get('name', ''))))
    
    except Exception as e:
        logger.error(f"Error listing instances: {e}", exc_info=True)
        return jsonify({"error": "Failed to list instances"}), 500



# UPDATED: Expects JSON { instance_name: "..." }
@app.route("/select_weaviate_instance", methods=["POST"])
def select_weaviate_instance():
    """Activates selected instance by updating RUNTIME config, stops other instances, and saves to YAML."""
    logger = app.logger
    try:
        data = request.get_json()
        instance_name = data.get('instance_name')
    except Exception as req_e: 
        return jsonify({"error": "Invalid request."}), 400
    
    if not instance_name: 
        return jsonify({"error": "Instance name required."}), 400
    
    if not config_available or not cfg: 
        return jsonify({"error": "Config system unavailable."}), 503
    
    selected_host, selected_http_port, selected_grpc_port = None, None, None
    
    try:
        # Get connection details for the selected instance
        if instance_name == "Default (from config)":
            # Reload from YAML
            with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f: 
                yaml_config = yaml.safe_load(f) or {}
            selected_host = yaml_config.get('retrieval', {}).get('WEAVIATE_HOST', 'localhost')
            try: selected_http_port = int(yaml_config.get('retrieval', {}).get('WEAVIATE_HTTP_PORT', 8080))
            except: selected_http_port = 8080
            try: selected_grpc_port = int(yaml_config.get('retrieval', {}).get('WEAVIATE_GRPC_PORT', 50051))
            except: selected_grpc_port = 50051
            logger.info(f"Selecting default instance from config: {selected_host}:{selected_http_port}")
        else:
            # Find in state file
            instance_state = load_weaviate_state()
            if instance_name not in instance_state: 
                return jsonify({"error": f"Instance '{instance_name}' not in state."}), 404
            
            details = instance_state[instance_name]
            selected_host = details.get('host')
            try: selected_http_port = int(details.get('http_port', -1))
            except: selected_http_port = -1
            try: selected_grpc_port = int(details.get('grpc_port', -1))
            except: selected_grpc_port = -1
            
            if not selected_host or selected_http_port == -1 or selected_grpc_port == -1:
                return jsonify({"error": f"Instance '{instance_name}' has invalid connection details."}), 400
            
            logger.info(f"Selecting managed instance '{instance_name}': {selected_host}:{selected_http_port}")
        
        # Stop all other instances regardless of which instance is being activated
        if docker_available:  # Remove the instance_name != "Default (from config)" condition
            instance_state = load_weaviate_state()
            for name, details in instance_state.items():
                if name != instance_name:  # Skip the instance we're activating
                    container_id = details.get("container_id")
                    if container_id:
                        try:
                            container = docker_client.containers.get(container_id)
                            if container.status == "running":
                                logger.info(f"Stopping inactive instance: {name}")
                                container.stop(timeout=30)  # Give it 30 seconds to gracefully stop
                                logger.info(f"Successfully stopped instance: {name}")
                                
                                # Update the instance state
                                instance_state[name]["status"] = "exited"
                        except DockerNotFound:
                            logger.warning(f"Container for instance '{name}' not found.")
                        except Exception as docker_err:
                            logger.error(f"Error stopping container for '{name}': {docker_err}")
            
            # Save the updated instance state
            save_weaviate_state(instance_state)

            # Start the selected instance if it's not running
            if instance_name != "Default (from config)":
                try:
                    selected_container_id = instance_state[instance_name].get("container_id")
                    if selected_container_id:
                        container = docker_client.containers.get(selected_container_id)
                        if container.status != "running":
                            logger.info(f"Starting selected instance: {instance_name}")
                            container.start()
                            logger.info(f"Successfully started instance: {instance_name}")
                            
                            # Update the instance state
                            instance_state[instance_name]["status"] = "running"
                            save_weaviate_state(instance_state)
                except Exception as start_err:
                    logger.error(f"Error starting selected instance '{instance_name}': {start_err}")
        
        # --- Update RUNTIME cfg object AND SAVE TO YAML ---
        cfg.retrieval.WEAVIATE_HOST = selected_host
        cfg.retrieval.WEAVIATE_HTTP_PORT = selected_http_port
        cfg.retrieval.WEAVIATE_GRPC_PORT = selected_grpc_port
        
        # Save the updated configuration to YAML
        try:
            with open(CONFIG_YAML_PATH, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            
            # Update the retrieval section with new values
            if 'retrieval' not in yaml_config:
                yaml_config['retrieval'] = {}
            
            yaml_config['retrieval']['WEAVIATE_HOST'] = selected_host
            yaml_config['retrieval']['WEAVIATE_HTTP_PORT'] = selected_http_port
            yaml_config['retrieval']['WEAVIATE_GRPC_PORT'] = selected_grpc_port
            
            # Write the updated config back to the file
            with open(CONFIG_YAML_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, indent=2, sort_keys=False)
            
            logger.info(f"Updated YAML config with '{instance_name}' as default instance")
        except Exception as yaml_err:
            logger.error(f"Failed to update YAML config: {yaml_err}")
            # Continue with runtime update even if YAML save fails
        
        logger.info(f"Runtime config updated for '{instance_name}'. Re-initializing pipeline...")
        
        # --- Re-initialize Pipeline ---
        initialize_pipeline(app.app_context())
        
        # --- Verify Connection ---
        if not pipeline or not pipeline.retriever or not pipeline.retriever.weaviate_client or not pipeline.retriever.weaviate_client.is_connected():
            logger.error(f"Pipeline failed to re-initialize/connect after selecting '{instance_name}'.")
            return jsonify({"error": "Failed to connect pipeline to selected instance."}), 500
        else:
            logger.info(f"Pipeline successfully connected to '{instance_name}'.")
            
            return jsonify({
                "success": True, 
                "message": f"Instance '{instance_name}' activated, set as default, and other instances stopped!", 
                "active_host": selected_host, 
                "active_http_port": selected_http_port
            })
    
    except Exception as e:
        logger.error(f"Error selecting instance '{instance_name}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to select instance: {e}"}), 500




# UPDATED: Expects JSON { instance_name: "..." }
@app.route('/remove_weaviate_instance', methods=['POST']) # Or DELETE if you prefer
def remove_weaviate_instance():
    if not docker_available:
        return jsonify({"success": False, "error": "Docker client unavailable."}), 503
    if not config_available or not cfg:
         return jsonify({"success": False, "error": "Configuration system unavailable."}), 503

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


# --- Make sure initialize_pipeline is called on startup ---

# Optional Debug route
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
    try:
        with open('auto_domain_keywords.txt', 'r') as file:
            keywords = file.read().strip()  # Read keywords from the file
            keyword_list = keywords.split(',')  # Split by comma to get an array
        return jsonify({'success': True, 'keywords': keyword_list})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# === Main Execution Block ===
if __name__ == '__main__':
    logger.info("Application starting...")
    if not config_available or not cfg: logger.critical("CRITICAL: Config failed load."); exit(1) # Using exit() directly here, consider sys.exit(1)

    try: # Setup directories early
        app.instance_path_obj = Path(app.instance_path); app.instance_path_obj.mkdir(parents=True, exist_ok=True)
        Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
        WEAVIATE_DATA_DIR_HOST.mkdir(parents=True, exist_ok=True)
        if cfg and cfg.paths and cfg.paths.DOCUMENT_DIR: Path(cfg.paths.DOCUMENT_DIR).resolve().mkdir(parents=True, exist_ok=True)
    except Exception as e: logger.error(f"CRITICAL: Dir creation failed: {e}"); exit(1) # Using exit() directly

    with app.app_context():
        logger.info("Entering application context for initialization.")
        try:
            # MODIFIED: Check if extensions are already initialized before calling init_app
            if not hasattr(db, 'app') or db.app is None:
                db.init_app(app)
            if not hasattr(sess, 'app') or sess.app is None:
                sess.init_app(app)
            logger.info("Flask extensions initialized.")
        except Exception as ext_e:
            logger.critical(f"Failed extensions init: {ext_e}", exc_info=True)
            exit(1) # Using exit() directly

        logger.info("Checking/Creating database tables...")
        try:
            db.create_all()
            logger.info("DB tables checked/created.")
        except Exception as db_e:
            logger.error(f"Failed DB create tables: {db_e}", exc_info=True)
            # Decide if you want to exit here or continue if DB creation fails

        # Initialize pipeline and load presets within the app context
        initialize_pipeline(app.app_context()) # Pass context explicitly if needed by the function
        load_presets() # Load global presets
        logger.info(f"Presets loaded in main block. Count: {len(presets)}") # Log preset count

    # --- Define and Register Cleanup Function ---
    import atexit
    # Note: Import weaviate here if not already globally imported, or move import inside function
    # import weaviate # Potentially move this import inside the function if not needed globally

    def cleanup_resources():
        # Close pipeline retriever client
        if 'pipeline' in globals() and pipeline:
            if hasattr(pipeline.retriever, 'close'):
                try:
                    pipeline.retriever.close()
                    logger.info("Closed pipeline retriever client")
                except Exception as e:
                    logger.error(f"Error closing retriever: {e}")

        # Close any other Weaviate clients
        try:
            logger.info("Attempting to close any remaining Weaviate connections")
            # Ensure weaviate is imported before use
            import weaviate
            # Find all client instances that might be open
            for client_attr in dir(weaviate):
                if client_attr.endswith('Client'):
                    client_class = getattr(weaviate, client_attr, None)
                    # Check if the class exists and has the _instances attribute
                    if client_class and hasattr(client_class, '_instances'):
                        # Iterate over a copy of the list as closing might modify it
                        for client in list(getattr(client_class, '_instances', [])):
                            try:
                                # Check if client has a close method and it's callable
                                if hasattr(client, 'close') and callable(client.close):
                                    client.close()
                                    logger.info(f"Closed {client_attr} instance")
                            except Exception as e:
                                logger.error(f"Error closing {client_attr} instance: {e}")
        except ImportError:
             logger.warning("Weaviate module not found during cleanup.")
        except Exception as e:
            logger.error(f"Error during additional Weaviate connection cleanup: {e}")

    atexit.register(cleanup_resources)
    # --- End Cleanup Function ---

    # --- Log Final Status Before Running ---
    if not pipeline: logger.warning("Pipeline initialization failed or skipped.")
    if not docker_available: logger.warning("Docker client unavailable.")
    logger.info(f"Flask-Session: Type={app.config.get('SESSION_TYPE')}, Dir={app.config.get('SESSION_FILE_DIR')}")
    logger.info(f"Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI')}")

    # --- ADD THE MISSING APP.RUN() CALL HERE ---
    logger.info(f"Starting Flask server on http://0.0.0.0:5000/ (Debug: {app.debug})")
    # Use debug=True carefully in production. use_reloader=False is often good for stability during dev.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
