# validate_configs.py
import sys
import json
import yaml
from pydantic import ValidationError
from config import AppConfig, SecurityConfig, RetrievalConfig, ModelConfig, \
                   DocumentConfig, PathConfig, EnvironmentConfig, \
                   PipelineConfig, IngestionConfig, DomainKeywordExtractionConfig

# Map section names to their Pydantic classes
SECTION_MODELS = {
    'security': SecurityConfig,
    'retrieval': RetrievalConfig,
    'model': ModelConfig,
    'document': DocumentConfig,
    'paths': PathConfig,
    'env': EnvironmentConfig,
    'pipeline': PipelineConfig,
    'ingestion': IngestionConfig,
    'domain_keyword_extraction': DomainKeywordExtractionConfig,
}

def main():
    errors = 0

    # 1) Validate config_settings.yaml
    try:
        AppConfig.load_from_yaml()
        print(" config_settings.yaml is valid")
    except ValidationError as e:
        print("config_settings.yaml validation errors:")
        print(e)
        errors += 1
    except Exception as e:
        print("config_settings.yaml load error:")
        print(e)
        errors += 1
        raw = {}

    # 2) Validate presets.json
    try:
            # safely open & load JSON, then close the file
        with open("presets.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("presets.json must be a JSON object at top level")
    except Exception as e:
        print(f"Could not read presets.json: {e}")
        sys.exit(1)

    preset_errors = 0
    for name, pdata in raw.items():
        if not isinstance(pdata, dict):
            print(f"Preset '{name}' must be an object, got {type(pdata).__name__}")
            preset_errors += 1
            continue

        # Warn & skip unknown sections (but still validate the rest)
        unknown = set(pdata) - set(SECTION_MODELS)
        if unknown:
            print(f"Preset '{name}' has unknown sections: {unknown} — skipping these")
            for key in unknown:
                pdata.pop(key, None)

        # Validate each section via its model
        for section, model_cls in SECTION_MODELS.items():
            if section in pdata:
                try:
                    model_cls(**pdata[section])
                except ValidationError as ve:
                    print(f"Preset '{name}' → section '{section}' errors:")
                    print(ve)
                    preset_errors += 1

    if preset_errors == 0:
        print("presets.json is valid")
    else:
        errors += preset_errors

    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
