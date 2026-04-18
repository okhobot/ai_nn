from config_manager import ConfigManager
cm=ConfigManager()
cm.load_config()
cm.set_hf_env()

from llama_cpp import Llama
import nn
from tts import TTS
from stt import STT
import re
import threading
import time
import subprocess
import json
import os
from memory_module import MemoryModule

class Talker:
    def __init__(self, tts):
        self.tts = tts
        self.text_to_speak = ""
    
    def set_text(self, text):
        self.text_to_speak = text
    
    def talk(self):
        while True:
            #print(": "+self.text_to_speak)
            if self.text_to_speak != "":
                text = self.text_to_speak
                self.text_to_speak = ""
                self.tts.speak_async(text)
                
            time.sleep(0.01)
    
    def stop(self):
        self.text_to_speak = ""
        self.tts.stop()

class Ai_NN:
    def __init__(self, json_config):
        self.text_to_say = ""
        self.mem_notes = []
        self.json_config=json_config
        
        self.neuro = nn.NN(
            repo_id=self.json_config["model"]["repo_id"], 
            filename=self.json_config["model"]["filename"],
            use_gpu=self.json_config["model"]["use_gpu"],
            save_history_count=2
            )

        self.tts = TTS(
            pitch_shift=self.json_config["tts"]["pitch_shift"],
            speaker=self.json_config["tts"]["speaker_name"],
            )

        self.stt = STT(
            call_func=self.on_input_text, 
            model_size=self.json_config["stt"]["model"],
            device=self.json_config["stt"]["device"],
            use_nr=self.json_config["stt"]["use_nr"]
            )
        
        self.mem_module = MemoryModule("config/mem_data.json")
                
        self.talker = Talker(self.tts)

        with open(self.json_config["model"]["init_prompt_path"], encoding="utf-8") as f:
            print(self.neuro.chat(f.read(), role=self.json_config["model"]["init_prompt_role"]))

    def run_powershell_command(self, command):
        """Выполняет команду PowerShell и возвращает результат"""
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                encoding='cp866'  # или 'utf-8', 'cp1251'
            )
            return result.stdout, result.stderr
        except Exception as e:
            return None, str(e)
            
    def split_com_blocks(self, text, com_name="powershell"):
        # Паттерн для поиска блоков ```powershell ... ```
        pattern = r'```' + com_name + '(.*?)```'
        
        # Находим все блоки
        blocks = re.findall(pattern, text, re.DOTALL)
        
        # Удаляем блоки из текста (получаем остальной текст)
        remaining = re.sub(pattern, '', text, flags=re.DOTALL)
        
        return remaining, blocks

    def on_input_text(self, text, depth=0):
        self.mem_notes = self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"])
        if len(self.mem_notes) > 0: text = text + "\nзаметки: \n" + "\n".join(self.mem_notes)
        print(">> " + text)
        self.tts.stop()
        res = self.neuro.chat(text)
        text, text_blocks = self.split_calls(res)
        
        print(res)
        print(text)

        if len(text) == 0:
            text = "ладно"
        
        #talker.set_text(res)
        text_to_say = text  # исправлено имя переменной
        self.tts.speak(text)

        self.process_calls(text_blocks, self.on_input_text, depth)

    def chat(self, text, depth=0):
        self.mem_notes = self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"])
        if len(self.mem_notes) > 0: text = text + "\nзаметки: \n" + "\n".join(self.mem_notes)
        print("<< " + text)
        res = self.neuro.chat(text)
        text, text_blocks = self.split_calls(res)
        
        print("responce:", res)
        print("text: ", text)
    
        self.process_calls(text_blocks, self.chat, depth)

    def split_calls(self, text):
        text, save_note = self.split_com_blocks(text, "_save")
        text, find_query = self.split_com_blocks(text, "_find")
        text, powershell = self.split_com_blocks(text)
        return text, (save_note, find_query, powershell)
    
    def process_calls(self, text_blocks, callback_func, depth):  # save_note, find_query, powershell
        if len(text_blocks[0]) > 0:
            for text in text_blocks[0]:
                self.mem_module.save(text)

        if len(text_blocks[1]) > 0:
            for text in text_blocks[1]:
                self.mem_notes += self.mem_module.request(text, self.json_config["model"]["load_embeddings_count"])

        if len(text_blocks[2]) > 0:
            cmd_out = ""
            for text in text_blocks[2]:
                cmd_out += "\n".join(self.run_powershell_command(text))
                print(cmd_out)
            cmd_out = cmd_out[:2000]
            if len(cmd_out) > 1 and depth < self.json_config["model"]["max_console_op_depth"]: 
                callback_func("вывод консоли: " + cmd_out, depth+1)

    def start_recognition(self):
        print("начало диалога")
        self.stt.start()
        func_thread = threading.Thread(target=self.talker.talk)
        func_thread.daemon = True
        func_thread.start()
        
    def stop_recognition(self):
        self.stt.stop()
    
    def calibrate(self, time):
        self.stt.calibrate(2)

    def get_text_to_say(self):
        return self.text_to_say

ai_nn = Ai_NN(cm.get_json_config())
if __name__ == "__main__":
    while True:
        ai_nn.calibrate(2)
        ai_nn.start_recognition()
        input()
        #ai_nn.chat(input(">>"))
        ai_nn.stop_recognition()