import os
import json

class ConfigManager:
    def __init__(self):
         self.json_config=None
    def load_config(self, path_to_config="config/config.json"):
        with open(path_to_config, 'r', encoding='utf-8') as f: 
            self.json_config = json.load(f)
    def set_hf_env(self):
            if self.json_config["cache_dir"] != "":
                cache_dir = self.json_config["cache_dir"]
                
                # Convert to absolute path if relative
                # Use the directory of the config file as base for relative paths
                if not os.path.isabs(cache_dir):
                    # Get the directory where the script is running from
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    cache_dir = os.path.join(script_dir, cache_dir)#, '..'
                    cache_dir = os.path.normpath(cache_dir)
                
                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)
                
                # CRITICAL: Set environment variables BEFORE importing huggingface_hub
                # This must happen first, before any huggingface_hub imports
                os.environ['HF_HOME'] = cache_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
                
                # PyTorch/Torch Hub models cache
                os.environ['TORCH_HOME'] = cache_dir
                os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = cache_dir
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                
                # Sentence Transformers cache
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
                
                # Faster Whisper cache
                os.environ['XDG_CACHE_HOME'] = cache_dir
                
                # General model download paths
                os.environ['MODELSCOPE_CACHE'] = cache_dir
                
                # NOW import and set constants (after env vars are set)
                import huggingface_hub.constants
                huggingface_hub.constants.HF_HOME = cache_dir
                huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = cache_dir
                
                print(f"[ConfigManager] Cache directory set to: {cache_dir}")
                print(f"[ConfigManager] HF_HOME constant: {huggingface_hub.constants.HF_HOME}")
                print(f"[ConfigManager] HUGGINGFACE_HUB_CACHE constant: {huggingface_hub.constants.HUGGINGFACE_HUB_CACHE}")
            else:
                print("[ConfigManager] No cache_dir specified, using default locations")
            
            token=self.json_config["hf_token"]
            os.environ['HF_TOKEN'] = token
            os.environ['HF_HUB_OFFLINE'] = str(int(self.json_config["offline"]))
            os.environ['TRANSFORMERS_OFFLINE'] = str(int(self.json_config["offline"]))
            #login(token=token, add_to_git_credential=False)
    
    def get_json_config(self):
         return self.json_config