import os
import json


class ConfigManager:
    def __init__(self):
        self.json_config = None
        
    def load_config(self, path_to_config="config/config.json"):
        """
        Load configuration from JSON file
        :param path_to_config: Path to the configuration file
        """
        with open(path_to_config, 'r', encoding='utf-8') as f: 
            self.json_config = json.load(f)
    
    def set_hf_env(self):
        """
        Set Hugging Face environment variables based on config
        """
        if self.json_config["cache_dir"] != "":
            os.environ['HF_HOME'] = self.json_config["cache_dir"]
        
        os.environ['HF_TOKEN'] = self.json_config["hf_token"]
        os.environ['HF_HUB_OFFLINE'] = str(int(self.json_config["offline"]))
        os.environ['TRANSFORMERS_OFFLINE'] = str(int(self.json_config["offline"]))
    
    def get_json_config(self):
        """
        Get the loaded JSON configuration
        :return: Loaded configuration dictionary
        """
        return self.json_config


# For backward compatibility when running as standalone
if __name__ == "__main__":
    cm = ConfigManager()
    cm.load_config()
    cm.set_hf_env()