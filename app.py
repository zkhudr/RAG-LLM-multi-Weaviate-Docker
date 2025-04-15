import os
# Disable optional telemetry if libraries like Weaviate use it
os.environ['POSTHOG_DISABLED'] = 'true'
import logging
import json
import traceback
import docker
from werkzeug.utils import secure_filename
import yaml
import csv
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_session import Session # Import Session from flask_session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from ingest_block import run_ingestion


# --- Configure Flask App ---
db = SQLAlchemy()
sess = Session()
# --- Configure Flask App ---
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key_for_dev_only') # Use env var or secure default
if app.secret_key == 'fallback_secret_key_for_dev_only':
    logging.warning("Using fallback Flask secret key. Set FLASK_SECRET_KEY environment variable for production.")

# --- Configure Database ---
db_path = os.path.join(app.instance_path, 'chat_history.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Configure Flask-Session ---
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False # Session expires when browser closes
app.config["SESSION_USE_SIGNER"] = True # Encrypt session cookie ID
app.config["SESSION_FILE_DIR"] = "./.flask_session" # Directory for session files
app.config["SESSION_FILE_THRESHOLD"] = 100 # Max number of session files before cleanup

##### start_weaviate_instance
client = docker.from_env()
weaviate_instances = {}

def start_weaviate_instance(instance_name):
    """
    Starts a new Weaviate container with the given instance name.
    This will also assign a unique port for each instance.
    """
    # Calculate a dynamic port for the new instance
    port = 8080 + len(weaviate_instances)

    # Run the Weaviate container with the specified port
    container = client.containers.run(
        "semitechnologies/weaviate",  # Docker image for Weaviate
        name=instance_name,
        ports={'8080/tcp': port},  # Bind to a dynamic port
        detach=True  # Run in detached mode
    )

    return container
##### 


class SavedChat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    history = db.Column(db.JSON)  # Store chat_history list as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- Import Project Modules with Error Handling ---
try:
    from pipeline import IndustrialAutomationPipeline
    pipeline_available = True
except ImportError:
    logging.error("Failed to import IndustrialAutomationPipeline. Check pipeline.py.", exc_info=True)
    pipeline_available = False
    IndustrialAutomationPipeline = None # Define as None

try:
    # Assuming config.py defines and loads 'cfg', provides AppConfig model and FlowStyleList
    from config import cfg, AppConfig, FlowStyleList
    config_available = True
except ImportError:
    logging.error("Failed to import configuration (cfg, AppConfig, FlowStyleList). Check config.py.", exc_info=True)
    config_available = False
    cfg = None # Define as None
    AppConfig = None
    FlowStyleList = list # Fallback

# Import Ingestion Components with Error Handling
try:
    from weaviate import connect_to_local # v4 client connection
    from ingest_docs_v7 import DocumentProcessor # Assuming this is your processor class name
    ingestion_available = True
except ImportError:
    logging.warning("Weaviate client or DocumentProcessor could not be imported. Ingestion will be unavailable.")
    connect_to_local = None
    DocumentProcessor = None
    ingestion_available = False

# Import Pydantic ValidationError (handle v1/v2 difference)
try:
    from pydantic import ValidationError # v2
except ImportError:
    try:
        from pydantic.error_wrappers import ValidationError # v1
    except ImportError:
        logging.error("Pydantic ValidationError could not be imported.")

        ValidationError = Exception # Fallback to base Exception

# --- Constants ---
CONFIG_YAML_PATH = Path("./config_settings.yaml").resolve() # Use Path object for robustness
PRESETS_FILE = Path("./presets.json").resolve()
try:
    os.makedirs(app.instance_path, exist_ok=True) # Ensure instance folder exists for SQLite
except OSError as e: logging.error(f"Could not create instance folder: {e}")

db.init_app(app)   # <<< ADD THIS LINE
sess.init_app(app)

# --- Ensure Session File Dir ---
try:
    Path(app.config["SESSION_FILE_DIR"]).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.error(f"Could not create session directory: {e}")
    
try:
    session_dir = Path(app.config["SESSION_FILE_DIR"])
    session_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.error(f"Could not create session directory '{app.config['SESSION_FILE_DIR']}': {e}")

# --- Initialize Pipeline ---
pipeline = None
if pipeline_available and config_available:
    try:
        # Pipeline initialization might depend on 'cfg' being loaded
        pipeline = IndustrialAutomationPipeline()
        logging.info("IndustrialAutomationPipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize IndustrialAutomationPipeline: {e}", exc_info=True)
else:
    logging.error("Cannot initialize pipeline due to missing dependencies or failed config load.")


# === Utility Functions (Define ONCE) ===

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
    """Loads presets from a JSON file."""
    if not filename.exists():
        logging.warning(f"Presets file '{filename}' not found. Returning empty presets.")
        return {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading presets from '{filename}': {e}")
        return {}

def save_presets(presets_data, filename=PRESETS_FILE):
    """Saves presets to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(presets_data, f, indent=2)
        logging.info(f"Presets saved to '{filename}'.")
    except IOError as e:
        logging.error(f"Failed to save presets to '{filename}': {e}")

def parse_form_to_config_dict(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parses Flask form data into a dictionary matching config structure."""
    # Use current cfg values as defaults where needed
    default_cfg_dict = cfg.model_dump() if cfg else {}

    # Helper to get nested defaults safely
    def get_default(keys, default_val=None):
        d = default_cfg_dict
        try:
            for key in keys:
                d = d[key]
            return d
        except (KeyError, TypeError):
            return default_val

    config_dict = {
        'security': {
            'SANITIZE_INPUT': 'SANITIZE_INPUT' in form_data,
            'CACHE_ENABLED': 'CACHE_ENABLED' in form_data,
            'RATE_LIMIT': safe_int(form_data.get('RATE_LIMIT'), default=get_default(['security', 'RATE_LIMIT'], 100), min_val=1),
            'API_TIMEOUT': safe_int(form_data.get('API_TIMEOUT'), default=get_default(['security', 'API_TIMEOUT'], 30), min_val=1),
            # API key is not managed via form
        },
        'retrieval': {
            'K_VALUE': safe_int(form_data.get('K_VALUE'), default=get_default(['retrieval', 'K_VALUE'], 6), min_val=1),
            'SEARCH_TYPE': form_data.get('SEARCH_TYPE') or get_default(['retrieval', 'SEARCH_TYPE'], 'mmr'),
            'SCORE_THRESHOLD': safe_float(form_data.get('SCORE_THRESHOLD'), default=get_default(['retrieval', 'SCORE_THRESHOLD'], 0.6), min_val=0, max_val=1),
            'LAMBDA_MULT': safe_float(form_data.get('LAMBDA_MULT'), default=get_default(['retrieval', 'LAMBDA_MULT'], 0.6), min_val=0, max_val=1),
            'PERFORM_DOMAIN_CHECK': 'PERFORM_DOMAIN_CHECK' in form_data,
            'DOMAIN_SIMILARITY_THRESHOLD': safe_float(form_data.get('DOMAIN_SIMILARITY_THRESHOLD'), default=get_default(['retrieval', 'DOMAIN_SIMILARITY_THRESHOLD'], 0.65), min_val=0, max_val=1),
            'SPARSE_RELEVANCE_THRESHOLD': safe_float(form_data.get('SPARSE_RELEVANCE_THRESHOLD'), default=get_default(['retrieval', 'SPARSE_RELEVANCE_THRESHOLD'], 0.15), min_val=0, max_val=1),
            'FUSED_RELEVANCE_THRESHOLD': safe_float(form_data.get('FUSED_RELEVANCE_THRESHOLD'), default=get_default(['retrieval', 'FUSED_RELEVANCE_THRESHOLD'], 0.45), min_val=0, max_val=1),
            'SEMANTIC_WEIGHT': safe_float(form_data.get('SEMANTIC_WEIGHT'), default=get_default(['retrieval', 'SEMANTIC_WEIGHT'], 0.7), min_val=0, max_val=1),
            'SPARSE_WEIGHT': safe_float(form_data.get('SPARSE_WEIGHT'), default=get_default(['retrieval', 'SPARSE_WEIGHT'], 0.3), min_val=0, max_val=1),
            # COLLECTION_NAME is often read-only/set internally
        },
        'model': {
            'OLLAMA_MODEL': (form_data.get('OLLAMA_MODEL') or get_default(['model', 'OLLAMA_MODEL'], '')).strip(),
            'EMBEDDING_MODEL': (form_data.get('EMBEDDING_MODEL') or get_default(['model', 'EMBEDDING_MODEL'], '')).strip(),
            'LLM_TEMPERATURE': safe_float(form_data.get('LLM_TEMPERATURE'), default=get_default(['model', 'LLM_TEMPERATURE'], 0.7), min_val=0),
            'MAX_TOKENS': safe_int(form_data.get('MAX_TOKENS'), default=get_default(['model', 'MAX_TOKENS'], 1024), min_val=1),
            'TOP_P': safe_float(form_data.get('TOP_P'), default=get_default(['model', 'TOP_P'], 1.0), min_val=0, max_val=1),
            'FREQUENCY_PENALTY': safe_float(form_data.get('FREQUENCY_PENALTY'), default=get_default(['model', 'FREQUENCY_PENALTY'], 0.0), min_val=0),
            'SYSTEM_MESSAGE': (form_data.get('SYSTEM_MESSAGE') or get_default(['model', 'SYSTEM_MESSAGE'], '')).strip(),
        },
        'document': {
            'CHUNK_SIZE': safe_int(form_data.get('CHUNK_SIZE'), default=get_default(['document', 'CHUNK_SIZE'], 1000), min_val=50),
            'CHUNK_OVERLAP': safe_int(form_data.get('CHUNK_OVERLAP'), default=get_default(['document', 'CHUNK_OVERLAP'], 100), min_val=0),
            'PARSE_TABLES': 'PARSE_TABLES' in form_data,
            'FILE_TYPES': form_data.getlist('FILE_TYPES') or get_default(['document', 'FILE_TYPES'], ['pdf']), # Use getlist for multiple checkboxes
        },
         'paths': {
            'DOCUMENT_DIR': (form_data.get('DOCUMENT_DIR') or get_default(['paths', 'DOCUMENT_DIR'], './data')).strip(),
            # Other paths usually not editable here
             'DOMAIN_CENTROID_PATH': get_default(['paths', 'DOMAIN_CENTROID_PATH'], './domain_centroid.npy') # Keep existing
        },
        'env': {
             'USER_ADDED_KEYWORDS': safe_split(form_data.get("USER_KEYWORDS", "")),
             # Keep non-editable env fields from current cfg for saving consistency
             'AUTO_DOMAIN_KEYWORDS': get_default(['env', 'AUTO_DOMAIN_KEYWORDS'], []),
             'DOMAIN_KEYWORDS': get_default(['env', 'DOMAIN_KEYWORDS'], []),
             'merged_keywords': get_default(['env', 'merged_keywords'], []) # Keep existing merged
        }
    }
    return config_dict

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

# --- Load Presets Initially ---
presets = load_presets()

# === Flask Routes ===
@app.route("/", methods=["GET", "POST"])

def index():
    """Handles main page display (GET) and config saving (POST)."""
    # Use app logger consistently
    logger = app.logger  # Use app's logger instance
    # Ensure config is available
    if not config_available or not cfg:
        logger.error("Config check failed: config_available=%s, cfg is None=%s", config_available, cfg is None)
        return render_template("index.html", config={}, presets={})  # Return minimal structure to avoid JS errors

    # --- POST Request Logic ---
    if request.method == "POST":
        try:
            config_updates_dict = parse_form_to_config_dict(request.form)
            config_changed = cfg.update_and_save(config_updates_dict)
            if config_changed:
                logger.info("Configuration updated via form POST.")
                # Optionally reload pipeline or trigger reinitialization logic here
            else:
                logger.info("No configuration changes detected from form POST.")
        except (ValidationError, ValueError) as e:
            logger.error(f"Config validation/save error on POST: {e}", exc_info=True)
            flash(f"Error saving config: Invalid values. {e}", "error")
        except Exception as e:
            logger.error(f"Config update failed on POST: {str(e)}", exc_info=True)
            flash(f"Error saving config: {str(e)}", "error")
        return redirect(url_for('index'))  # Redirect after POST

    # --- GET Request Logic ---
    logger.info(f"--- Preparing GET / response ---")

    presets_reloaded = {}
    config_dict_for_template = {}

    try:
        # Load Presets Safely
        presets_reloaded = load_presets()
        logger.info(f"Loaded presets: {list(presets_reloaded.keys())}")

        # Prepare Config Dict Safely
        dump_method = getattr(cfg, 'model_dump', getattr(cfg, 'dict', None))

        if dump_method:
            logger.debug(f"Attempting config dump using {dump_method.__name__}")
            try:
                config_dict_for_template = dump_method(exclude={
                    'security': {'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'}
                })
                logger.debug("Config dump successful.")

                # Add boolean flags for API keys in the security section
                sec_data = config_dict_for_template.setdefault('security', {})
                sec_data['DEEPSEEK_API_KEY'] = bool(getattr(cfg.security, 'DEEPSEEK_API_KEY', ''))
                sec_data['OPENAI_API_KEY'] = bool(getattr(cfg.security, 'OPENAI_API_KEY', ''))
                sec_data['ANTHROPIC_API_KEY'] = bool(getattr(cfg.security, 'ANTHROPIC_API_KEY', ''))
                sec_data['COHERE_API_KEY'] = bool(getattr(cfg.security, 'COHERE_API_KEY', ''))
                logger.debug("Added API key boolean flags.")
            except AttributeError as attr_err:
                logger.error(f"Error preparing config dict: {attr_err}", exc_info=True)
                config_dict_for_template = {'security': {}, 'retrieval': {}, 'model': {}, 'document': {}, 'paths': {}, 'env': {}}
                logger.warning("Using empty config structure due to error.")
            except Exception as e:
                logger.error(f"Unexpected error preparing config: {e}", exc_info=True)
                config_dict_for_template = {}

        else:
            logger.error("Config method (.model_dump or .dict) not found on cfg object.")
            config_dict_for_template = {}

    except Exception as e:
        logger.error(f"Error during GET data preparation: {e}", exc_info=True)
        presets_reloaded = {}
        config_dict_for_template = {}
        print("Config being passed to template:", config_dict_for_template)

    #logger.info(f"Final config dict: {config_dict_for_template}")
    #logger.info(f"Final presets dict: {presets_reloaded}")

    # Return the rendered template with config and presets
    return render_template(
        "index.html",
        config=config_dict_for_template,  # Pass config to template
        presets=presets_reloaded         # Pass presets to template
    )



@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    """Handles chat message processing."""
    if not pipeline: # Check if pipeline loaded successfully
         return jsonify({
            "role": "assistant",
            "content": "Error: The question processing pipeline is not available.",
            "error": True,
            "timestamp": datetime.now().isoformat()
            }), 503 # Service Unavailable

    try:
        user_query = request.form.get("query", "").strip()
        if not user_query:
            # Return specific error message, don't add to history
             return jsonify({
                "role": "assistant",
                "content": "Please enter a query.",
                "error": True, # Indicate it's not a real response
                "timestamp": datetime.now().isoformat()
            }), 400 # Bad Request

        # 1. Get chat history from session
        chat_history = session.get('chat_history', [])

        # 2. Append new user query
        chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })

        # 3. Call pipeline with history
        # The updated pipeline.py MUST handle the chat_history argument
        result_dict = pipeline.generate_response(query=user_query, chat_history=chat_history)

        # 4. Append assistant response to history
        # Use "response" key from pipeline's return value for content
        assistant_response_content = result_dict.get("response", "Sorry, I couldn't generate a response.")
        assistant_message = {
            "role": "assistant",
            "content": assistant_response_content, # Use "content" key for frontend
            "source": result_dict.get("source", "unknown"),
            "model": result_dict.get("model", "unknown"),
            "context": result_dict.get("context"), # Pass context if pipeline returns it
            "timestamp": result_dict.get("timestamp", datetime.now().isoformat()),
            "error": result_dict.get("error", False) # Pass error flag if pipeline sets it
        }
        chat_history.append(assistant_message)

        # 5. Save updated history back to session
        session['chat_history'] = chat_history

        # 6. Return only the NEW assistant message for the frontend chat UI
        return jsonify(assistant_message)

    except Exception as e:
        app.logger.error(f"Error in /run_pipeline: {str(e)}", exc_info=True)
        # Return error message in the standard chat format
        return jsonify({
            "role": "assistant",
            "content": f"Sorry, an internal error occurred: {str(e)}",
            "error": True,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/apply_preset/<preset_name>", methods=["POST"])
def apply_preset(preset_name):
    """Applies a saved preset and returns the updated config via JSON."""
    global presets
    presets = load_presets()
    if preset_name not in presets: return jsonify({"error": "Preset not found"}), 404
    if not config_available or not cfg: return jsonify({"error": "Config system unavailable."}), 500

    try:
        preset_data = presets[preset_name]
        config_changed = cfg.update_and_save(preset_data) # Apply and save

        if config_changed:
             logger.info(f"Preset '{preset_name}' applied successfully. Configuration changed.")
# Reload pipeline HERE if immediate sync is mandatory, otherwise omit for faster UI respons
             global pipeline; 
             if pipeline_available: 
                 try: pipeline = IndustrialAutomationPipeline(); logger.info("Pipeline reloaded.")
                 except Exception as e: logger.error(...)
            #Return success + new config state (excluding sensitive keys)
             return jsonify({
                 "success": True,
                 "config": cfg.model_dump(exclude={'security': { # Ensure all keys listed
                     'DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY'
                     }})
             })
        else:
            logger.info(f"Preset '{preset_name}' applied, no configuration changes detected.")
            # Return success, indicating no frontend config update needed
            return jsonify({"success": True, "config": None})

    except (ValidationError, ValueError) as e:
        # ... (keep existing validation error handling) ...
        app.logger.error(f"Preset validation error applying '{preset_name}': {e}", exc_info=True)
        details = str(e)
        if isinstance(e, ValidationError): details = ". ".join([f"{err.get('loc', ['config'])[0]}: {err.get('msg', '')}" for err in e.errors()])
        return jsonify({"error": f"Invalid preset values: {details}"}), 400
    except Exception as e:
        # ... (keep existing general error handling) ...
        app.logger.error(f"Preset application error '{preset_name}': {e}", exc_info=True)
        return jsonify({"error": "Error applying preset"}), 500



@app.route("/save_preset", methods=["POST"])
def save_preset():
    """Saves the current form settings as a new preset."""
    global presets
    preset_name = request.form.get("preset_name", "").strip()
    if not preset_name:
         return jsonify({"error": "Preset name cannot be empty"}), 400

    if not config_available:
        return jsonify({"error": "Configuration system not available."}), 500

    try:
        # Capture current form settings into a dictionary
        preset_data = parse_form_to_config_dict(request.form)

        # Validate the captured data (optional but recommended)
        try:
             _ = AppConfig(**preset_data) # Attempt validation
        except ValidationError as ve_preset:
             app.logger.error(f"Preset data validation failed: {ve_preset}")
             raise ValueError(f"Invalid values captured for preset: {ve_preset}") # Raise clearer error

        # Save the validated preset data to the JSON file
        presets = load_presets() # Reload before updating
        presets[preset_name] = preset_data
        save_presets(presets)

        # Don't automatically apply it here, just save it. Applying is separate.
        # flash(f"Preset '{preset_name}' saved successfully!", "success") # Use JSON response instead
        # return redirect(url_for('index'))
        return jsonify({"success": True, "message": f"Preset '{preset_name}' saved."})

    except (ValueError, ValidationError) as e: # Catch validation errors specifically
        app.logger.error(f"Validation failed during preset save: {e}")
        return jsonify({"error": f"Invalid configuration values: {e}"}), 400
    except Exception as e:
        app.logger.error(f"Preset save failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to save preset: {str(e)}"}), 500

@app.route("/update_auto_keywords", methods=["POST"])
def update_auto_keywords():
    """Updates auto keywords from file and saves config."""
    if not config_available or not cfg: return jsonify({"error": "Config system unavailable."}), 500
    try:
        new_keywords_file = "new_auto_keywords.txt"
        new_auto_keywords = get_new_auto_keywords(new_keywords_file) # Use helper

        # --- Archiving logic (Optional) ---
        # ...

        # <<< CHANGE: Construct update dict and use cfg.update_and_save() >>>
        update_dict = {'env': {'AUTO_DOMAIN_KEYWORDS': new_auto_keywords}}
        config_changed = cfg.update_and_save(update_dict)
        # <<< END CHANGE >>>

        # Archiving or other post-save actions could go here

        # No need to reload pipeline just for keywords usually
        if config_changed:
             return jsonify({"success": True, "message": "Auto keywords updated successfully!"})
        else:
             return jsonify({"success": True, "message": "Auto keywords file processed, no changes detected."}) # Or just success

    except FileNotFoundError as fnf:
         app.logger.error(f"Update auto keywords failed: {fnf}")
         return jsonify({"error": str(fnf)}), 404
    except (ValidationError, ValueError) as e: # Catch potential validation errors if update_and_save raises them
        app.logger.error(f"Validation error updating auto keywords: {e}", exc_info=True)
        return jsonify({"error": f"Invalid keyword data: {e}"}), 400
    except Exception as e:
        app.logger.error(f"Update auto keywords failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/upload_files', methods=['POST'])
def upload_files():
    """Handles file uploads to the document directory."""
    if not config_available:
        return jsonify({"success": False, "error": "Configuration system not available."}), 500
    try:
        upload_dir = Path(cfg.paths.DOCUMENT_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as dir_e:
         app.logger.error(f"Upload directory error: {dir_e}", exc_info=True)
         return jsonify({"success": False, "error": f"Upload directory error: {dir_e}"}), 500

    try:
        files = request.files.getlist("files")
        if not files or all(not f or f.filename == '' for f in files):
            return jsonify({"success": False, "error": "No files selected"}), 400

        saved_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                save_path = upload_dir / filename
                file.save(save_path)
                saved_files.append(filename)
                app.logger.info(f"Uploaded file: {save_path}")

        return jsonify({"success": True, "files": saved_files})
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"File upload failed: {str(e)}"}), 500

@app.route('/start_ingestion', methods=['POST'])
def start_ingestion():
    """Triggers the document ingestion process."""
    if not ingestion_available:
         return jsonify({'success': False, 'error': 'Ingestion components not available'}), 503
    if not config_available:
         return jsonify({'success': False, 'error': 'Configuration system not available.'}), 500

    weaviate_client = None
    try:
        # Get Weaviate connection details from config
        # Adapt keys if your config structure is different
        host = getattr(cfg.retrieval, 'WEAVIATE_HOST', 'localhost')
        http_port = getattr(cfg.retrieval, 'WEAVIATE_HTTP_PORT', 8080)
        grpc_port = getattr(cfg.retrieval, 'WEAVIATE_GRPC_PORT', 50051)

        app.logger.info(f"Connecting to Weaviate at {host}:{http_port} for ingestion...")
        # Use connect_to_local assuming standard setup
        weaviate_client = connect_to_local(host=host, port=http_port, grpc_port=grpc_port)

        if not hasattr(weaviate_client, 'is_connected') or not weaviate_client.is_connected():
             raise ConnectionError("Failed to connect Weaviate for ingestion (check host/ports).")
        app.logger.info("Weaviate connected.")

        # Get document directory from form (if sent) or config
        doc_dir_str = request.form.get('document_dir', cfg.paths.DOCUMENT_DIR)
        doc_dir = Path(doc_dir_str).resolve() # Resolve to absolute path
        if not doc_dir.is_dir():
             raise FileNotFoundError(f"Ingestion source directory not found: {doc_dir}")

        app.logger.info(f"Starting ingestion process for directory: {doc_dir}")
        # Ensure DocumentProcessor __init__ matches arguments provided
        processor = DocumentProcessor(data_dir=doc_dir, client=weaviate_client)
        processor.execute() # Execute the ingestion

        stats = getattr(processor, 'get_stats', lambda: {})() # Get stats safely
        app.logger.info(f"Ingestion process finished. Stats: {stats}")
        return jsonify({
            'success': True,
            'message': 'Documents ingested successfully',
            'stats': stats
        })
    except FileNotFoundError as fnf_e:
        app.logger.error(f"Ingestion failed: {fnf_e}")
        return jsonify({'success': False, 'error': str(fnf_e)}), 404
    except ConnectionError as conn_e:
        app.logger.error(f"Ingestion failed: {conn_e}")
        return jsonify({'success': False, 'error': str(conn_e)}), 503 # Service Unavailable
    except Exception as e:
        app.logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Ingestion process failed: {str(e)}",
            'traceback': traceback.format_exc() # Optional: for detailed client-side debug
        }), 500
    finally:
        # Ensure client is closed if it was opened
        if weaviate_client and hasattr(weaviate_client, 'close') and callable(weaviate_client.close):
            try:
                weaviate_client.close()
                app.logger.info("Weaviate client closed after ingestion attempt.")
            except Exception as close_e:
                app.logger.error(f"Error closing Weaviate client: {close_e}")

@app.route("/ingest_block", methods=["POST"])
def ingest_block_route():
    try:
        # Get the document folder from the POST form data, or default to the one defined in cfg.paths.DOCUMENT_DIR.
        document_dir = request.form.get("document_dir", None)
        if not document_dir:
            document_dir = cfg.paths.DOCUMENT_DIR
        
        # Call the incremental ingestion function.
        result = run_ingestion(document_dir)
        
        return jsonify({
            "success": True,
            "message": result.get("message"),
            "stats": result
        })
    except Exception as e:
        app.logger.error("Error during incremental ingestion: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500                

@app.route('/save_chat', methods=['POST'])
def save_chat():
    """Saves the current session's chat history."""
    if 'chat_history' not in session or not session['chat_history']:
        return jsonify({"success": False, "error": "No chat history in session to save."}), 400
    data = request.get_json()
    chat_name = data.get('name', '').strip()
    if not chat_name:
        return jsonify({"success": False, "error": "Chat name is required."}), 400

    try:
        new_chat = SavedChat(name=chat_name, history=session['chat_history'])
        db.session.add(new_chat)
        db.session.commit()
        logger.info(f"Saved chat '{chat_name}' with ID {new_chat.id}")
        # Optionally return the new list of chats or just success
        return jsonify({"success": True, "id": new_chat.id, "name": new_chat.name})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving chat: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500

@app.route('/load_chat/<int:chat_id>', methods=['GET']) # Use GET for loading
def load_chat(chat_id):
    """Loads a saved chat history into the current session."""
    try:
        chat = db.session.get(SavedChat, chat_id) # Use db.session.get for primary key lookup
        if chat:
            session['chat_history'] = chat.history # Load into current session
            logger.info(f"Loaded chat '{chat.name}' (ID: {chat_id}) into session.")
            return jsonify({"success": True, "history": chat.history})
        else:
            return jsonify({"success": False, "error": "Chat not found."}), 404
    except Exception as e:
        logger.error(f"Error loading chat ID {chat_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500

@app.route('/list_chats', methods=['GET'])
def list_chats():
    """Lists all saved chats."""
    try:
        chats = SavedChat.query.order_by(SavedChat.created_at.desc()).all()
        return jsonify([{"id": c.id, "name": c.name} for c in chats])
    except Exception as e:
        logger.error(f"Error listing chats: {e}", exc_info=True)
        return jsonify({"error": f"Database error: {e}"}), 500

# === Optional: Route Listing for Debugging ===
@app.route("/list_routes")
def list_routes():
    """List all registered routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": sorted([m for m in rule.methods if m not in ('HEAD', 'OPTIONS')]), # Filter methods
            "path": str(rule)
        })
    return jsonify({"routes": sorted(routes, key=lambda x: x["path"])})


######################## Multi weaviate instances #########################
weaviate_instances = {}

@app.route("/create_weaviate_instance", methods=["POST"])
def create_weaviate_instance():
    """Creates a new Weaviate instance."""
    instance_name = request.json.get('instance_name')
    if instance_name in weaviate_instances:
        return jsonify({"error": "Instance name already exists!"}), 400

    # Dynamically create the instance (modify docker-compose.yml or start a container)
    # For simplicity, we're just storing the instance in a dictionary, 
    # but in a real-world scenario, you'd start a Docker container here.

    weaviate_instances[instance_name] = {
        "status": "created",
        "url": f"http://localhost:{8080 + len(weaviate_instances)}"  # Example: increment port number
    }

    return jsonify({"success": True, "message": f"Instance {instance_name} created successfully!"})


@app.route("/list_weaviate_instances", methods=["GET"])
def list_weaviate_instances():
    """Returns a list of active Weaviate instances."""
    return jsonify(list(weaviate_instances.keys()))

@app.route("/select_weaviate_instance", methods=["POST"])
def select_weaviate_instance():
    """Activate a selected Weaviate instance for RAG."""
    instance_name = request.json.get('instance_name')

    if instance_name not in weaviate_instances:
        return jsonify({"error": "Instance does not exist!"}), 400

    # Activate the instance by updating the configuration (e.g., Weaviate client, model settings)
    selected_instance = weaviate_instances[instance_name]
    # Here, update your system to use the selected Weaviate instance (e.g., update the client URL)
    
    return jsonify({"success": True, "message": f"Instance {instance_name} selected for RAG!"})


################################################################
# === Main Execution Block (Corrected db.create_all call) ===
if __name__ == '__main__':
    # 1. Configure Logging First
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Application starting...")

    # 2. CRITICAL CHECK: Configuration
    if not config_available or not cfg:
         logger.critical("CRITICAL: Configuration failed to load. Application cannot start.")
         exit(1)

    # 3. Perform Setup Tasks (Upload Dir)
    logger.info(f"Config loaded successfully. Document dir: {cfg.paths.DOCUMENT_DIR}")
    try:
        Path(cfg.paths.DOCUMENT_DIR).resolve().mkdir(parents=True, exist_ok=True)
        logger.info("Checked/Ensured upload directory.")
    except Exception as dir_e:
        logger.error(f"Could not create upload directory: {dir_e}")

    # Create Database Tables
    # Calling db.create_all() within app_context is correct
    with app.app_context():
        logger.info("Checking/Creating database tables...")
        try:
            # --- CORRECTED LINE ---
            db.create_all() # Call without arguments
            # --- END CORRECTED LINE ---
            logger.info("Database tables checked/created.")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}", exc_info=True)
            # Decide if this is fatal

    # 4. Check Non-Critical Components
    if not pipeline:
        logger.warning("Pipeline not initialized successfully.")

    # 5. Log Final Info and Run App
    logger.info(f"Flask-Session: Type={app.config.get('SESSION_TYPE')}, Dir={app.config.get('SESSION_FILE_DIR')}")
    logger.info(f"Database URI: {app.config.get('SQLALCHEMY_DATABASE_URI')}")
    logger.info(f"Starting Flask server on http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)



 
   

