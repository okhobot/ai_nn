import os
import json
from huggingface_hub import login, interpreter_login

class ConfigManager:
    def __init__(self):
         self.json_config=None
    def load_config(self, path_to_config="config/config.json"):
        with open(path_to_config, 'r', encoding='utf-8') as f: 
            self.json_config = json.load(f)
    def set_hf_env(self):
            if self.json_config["cache_dir"] != "":
                os.environ['HF_HOME'] = self.json_config["cache_dir"]
            
            token=self.json_config["hf_token"]
            os.environ['HF_TOKEN'] = token
            os.environ['HF_HUB_OFFLINE'] = str(int(self.json_config["offline"]))
            os.environ['TRANSFORMERS_OFFLINE'] = str(int(self.json_config["offline"]))
            #login(token=token, add_to_git_credential=False)
    
    def get_json_config(self):
         return self.json_config