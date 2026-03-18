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


#neuro=nn.NN("google/gemma-3-4b-it-qat-q4_0-gguf","gemma-3-4b-it-q4_0.gguf")
#neuro=nn.NN("google/gemma-3-1b-it-qat-q4_0-gguf","gemma-3-1b-it-q4_0.gguf") 
#neuro=nn.NN("Vikhrmodels/Vikhr-Qwen-2.5-0.5B-instruct-GGUF", "Vikhr-Qwen-2.5-0.5B-instruct-Q8_0.gguf")#
#neuro=nn.NN("gvij/qwen3-0.6b-gguf", "qwen3-0.6b-q8_0.gguf")
class Talker:
    text_to_speak=""
    def set_text(self,text):
        self.text_to_speak=text
    def talk(self):
        while True:
            #print(": "+self.text_to_speak)
            if self.text_to_speak!="":
                text=self.text_to_speak
                self.text_to_speak=""
                tts.speak_async(text)
                
            time.sleep(0.1)
def run_powershell_command(command):
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
    
def split_powershell_blocks(text):
    # Паттерн для поиска блоков ```powershell ... ```
    pattern = r'```powershell(.*?)```'
    
    # Находим все блоки
    blocks = re.findall(pattern, text, re.DOTALL)
    
    # Удаляем блоки из текста (получаем остальной текст)
    remaining = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return remaining, blocks

def on_input_text(text, depth=0):
    print(">> "+text)
    res=neuro.chat(text)
    text, powershell=split_powershell_blocks(res)
    print(res)
    print(powershell)

    if len(text)==0:
        text="ладно"
    
    #talker.set_text(res)
    tts.speak(text)

    if len(powershell)>0:
        cmd_out=run_powershell_command(powershell[0])
        print(cmd_out)
        if len(cmd_out[0]+cmd_out[1])>0 and depth<json_config["model"]["max_console_op_depth"]: on_input_text("вывод консоли: "+cmd_out[0]+"\n"+cmd_out[1], depth+1)
    
json_config=None
with open('config/config.json', 'r', encoding='utf-8') as f: json_config = json.load(f)
if json_config["cache_dir"]=="":json_config["cache_dir"]=None

neuro=nn.NN(
    repo_id= json_config["model"]["repo_id"], 
    filename= json_config["model"]["filename"],
    cache_dir=json_config["cache_dir"],
    hf_token= json_config["hf_token"],
    use_gpu=True, 
    offline=json_config["offline"],
    save_history_count=2
    )

tts=TTS(
    pithc_shift= json_config["tts"]["pitch_shift"],
    speaker= json_config["tts"]["speaker_name"],
    cache_dir=json_config["cache_dir"],
    offline=json_config["offline"]
    )

stt=STT(
    call_func=on_input_text, 
    model_size= json_config["stt"]["model"],
    device=json_config["stt"]["device"],
    cache_dir=json_config["cache_dir"], 
    use_nr=json_config["stt"]["use_nr"]
    )

talker=Talker()
with open(json_config["model"]["init_prompt_path"], encoding="utf-8") as f:
    print(neuro.chat(f.read(),role=json_config["model"]["init_prompt_role"]))

stt.calibrate(2)
print("начало диалога")
stt.start()
func_thred = threading.Thread(target=talker.talk)
func_thred.daemon = True
func_thred.start()
input()
stt.stop()
