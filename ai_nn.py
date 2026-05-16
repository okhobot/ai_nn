import re
import threading
import time
import subprocess
import json
import os

# Handle imports with fallback for both package and script usage
try:
    # Try relative imports when used as a package
    from .config_manager import ConfigManager
    from .nn import NN
    from .tts import TTS
    from .stt import STT
    from .memory_module import MemoryModule
    USING_RELATIVE_IMPORTS = True
except ImportError:
    # Fall back to absolute imports when run as a script
    from config_manager import ConfigManager
    from nn import NN
    from tts import TTS
    from stt import STT
    from memory_module import MemoryModule
    USING_RELATIVE_IMPORTS = False


class Talker:
    def __init__(self, tts):
        """
        Initialize the talker
        :param tts: TTS instance
        """
        self.tts = tts
        self.text_to_speak = ""
    
    def set_text(self, text):
        """
        Set text to speak
        :param text: Text to speak
        """
        self.text_to_speak = text
    
    def talk(self):
        """
        Main loop to handle speaking
        """
        while True:
            if self.text_to_speak != "":
                text = self.text_to_speak
                self.text_to_speak = ""
                self.tts.speak_async(text)
                
            time.sleep(0.01)
    
    def stop(self):
        """
        Stop current speech
        """
        self.text_to_speak = ""
        self.tts.stop()


class Ai_NN:
    def __init__(self, json_config):
        """
        Initialize the AI Neural Network assistant
        :param json_config: Configuration dictionary
        """
        self.text_to_speak = ""
        self.mem_notes = []
        self.json_config = json_config

        self.chat_history_path = self.json_config["model"]["chat_history_path"]
        if len(self.chat_history_path) == 0: 
            self.chat_history_path = None
        
        self.neuro = NN(
            repo_id=self.json_config["model"]["repo_id"], 
            filename=self.json_config["model"]["filename"],
            use_gpu=self.json_config["model"]["use_gpu"],
            save_history_count=self.json_config["model"]["chat_size"],
            history_file_path=self.chat_history_path
        )

        self.tts = TTS(
            pitch_shift=self.json_config["tts"]["pitch_shift"],
            speaker=self.json_config["tts"]["speaker_name"]
        )

        self.stt = STT(
            call_func=self.on_input_text, 
            model_size=self.json_config["stt"]["model"],
            device=self.json_config["stt"]["device"],
            use_nr=self.json_config["stt"]["use_nr"],
            device_index=self.json_config["stt"]["micro_index"],
            silence_duration=self.json_config["stt"]["silence_duration"]
        )
        
        self.mem_module = MemoryModule("config/mem_data.json")
                
        self.talker = Talker(self.tts)

        with open(self.json_config["model"]["init_prompt_path"], encoding="utf-8") as f:
            print(self.neuro.chat(f.read(), role=self.json_config["model"]["init_prompt_role"], save_chat=False))

    def load_history(self, history_file_path=None):
        """
        Load chat history
        :param history_file_path: Path to history file
        """
        self.neuro.load_history(history_file_path)

    def remove_history(self):
        """
        Remove chat history file
        """
        try:
            if self.chat_history_path and os.path.exists(self.chat_history_path):
                os.remove(self.chat_history_path)
        except Exception as e:
            print(f"Error removing history: {e}")

    def run_powershell_command(self, command):
        """
        Execute a PowerShell command and return the result
        :param command: Command to execute
        :return: Tuple of (stdout, stderr)
        """
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                encoding='cp866'  # or 'utf-8', 'cp1251'
            )
            return result.stdout, result.stderr
        except Exception as e:
            return None, str(e)
            
    def split_com_blocks(self, text, com_name="powershell"):
        """
        Split command blocks from text
        :param text: Input text
        :param com_name: Command name to look for
        :return: Tuple of (remaining text, found blocks)
        """
        # Pattern to find blocks like ```powershell ... ```
        pattern = r'```' + com_name + r'(.*?)```'
        
        # Find all blocks
        blocks = re.findall(pattern, text, re.DOTALL)
        
        # Remove blocks from text (get remaining text)
        remaining = re.sub(pattern, '', text, flags=re.DOTALL)
        
        return remaining, blocks

    def on_input_text(self, text, depth=0):
        """
        Handle input text from STT
        :param text: Input text
        :param depth: Recursion depth
        """
        self.mem_notes = self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"])
        if len(self.mem_notes) > 0: 
            text = text + "\nзаметки: \n" + "\n".join(self.mem_notes)
        print(">> " + text)
        self.tts.stop()
        res = self.neuro.chat(text)
        text, text_blocks = self.split_calls(res)
        
        print(res)
        print(text)

        if len(text) == 0:
            text = "ладно"
        
        text_to_say = text  # Fixed variable name
        self.tts.speak(text)

        self.process_calls(text_blocks, self.on_input_text, depth)

    def chat(self, text, depth=0):
        """
        Chat with the AI
        :param text: Input text
        :param depth: Recursion depth
        """
        self.mem_notes = self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"])
        if len(self.mem_notes) > 0: 
            text = text + "\nзаметки: \n" + "\n".join(self.mem_notes)
        print("<< " + text)
        res = self.neuro.chat(text)
        text, text_blocks = self.split_calls(res)
        
        print("Response:", res)
        print("Text: ", text)
    
        self.process_calls(text_blocks, self.chat, depth)

    def split_calls(self, text):
        """
        Split text into different command blocks
        :param text: Input text
        :return: Tuple of (clean text, command blocks)
        """
        text, save_note = self.split_com_blocks(text, "_save")
        text, find_query = self.split_com_blocks(text, "_find")
        text, powershell = self.split_com_blocks(text)
        return text, (save_note, find_query, powershell)
    
    def process_calls(self, text_blocks, callback_func, depth):  
        """
        Process command blocks
        :param text_blocks: Command blocks to process
        :param callback_func: Callback function
        :param depth: Recursion depth
        """
        # Handle save notes
        if len(text_blocks[0]) > 0:
            for text in text_blocks[0]:
                self.mem_module.save(text)

        # Handle find queries
        if len(text_blocks[1]) > 0:
            for text in text_blocks[1]:
                self.mem_notes.extend(self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"]))

        # Handle PowerShell commands
        if len(text_blocks[2]) > 0:
            cmd_out = ""
            for text in text_blocks[2]:
                stdout, stderr = self.run_powershell_command(text)
                cmd_out += stdout or ""
                if stderr:
                    cmd_out += f"\nError: {stderr}"
                print(cmd_out)
            cmd_out = cmd_out[:2000]
            if len(cmd_out) > 1 and depth < self.json_config["model"]["max_console_op_depth"]: 
                callback_func("вывод консоли: " + cmd_out, depth+1)

    def start_recognition(self):
        """
        Start speech recognition
        """
        print("Начало диалога")
        self.stt.start()
        func_thread = threading.Thread(target=self.talker.talk)
        func_thread.daemon = True
        func_thread.start()
        
    def stop_recognition(self):
        """
        Stop speech recognition
        """
        self.stt.stop()
    
    def calibrate(self, duration):
        """
        Calibrate STT
        :param duration: Calibration duration
        """
        self.stt.calibrate(duration)

    def get_text_to_say(self):
        """
        Get text to say
        :return: Current text to say
        """
        return self.text_to_speak


def main():
    """
    Main entry point for the AI-NN package when run as a script
    """
    cm = ConfigManager()
    cm.load_config()
    cm.set_hf_env()
    
    ai_nn = Ai_NN(cm.get_json_config())
    ai_nn.remove_history()
    ai_nn.load_history()
    #while True:
    #    ai_nn.chat(input(">>"))
    ai_nn.start_recognition()
    input()
    ai_nn.stop_recognition()


# Allow the module to be run as a script
if __name__ == "__main__":
    main()