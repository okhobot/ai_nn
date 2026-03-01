from llama_cpp import Llama
import nn
from tts import TTS
from stt import STT
import re
import threading
import time
import subprocess
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
    pattern = r'```powershell\n(.*?)```'
    
    # Находим все блоки
    blocks = re.findall(pattern, text, re.DOTALL)
    
    # Удаляем блоки из текста (получаем остальной текст)
    remaining = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return remaining, blocks

def on_input_text(text):
    print(">> "+text)
    res=neuro.chat(text)
    text, powershell=split_powershell_blocks(res)
    print(res)
    print(powershell)
    if len(powershell)>0:
        print(run_powershell_command(powershell[0]))
    #print(type(text))
    if len(text)==0:
        text="ок"
    
    #talker.set_text(res)
    tts.speak(text)

neuro=nn.NN("tiiuae/Falcon-H1-1.5B-Instruct-GGUF", "Falcon-H1-1.5B-Instruct-Q4_0.gguf","config/hf_token.txt",True)
tts=TTS(5,"kseniya")
stt=STT(on_input_text, "small","cpu",600)
talker=Talker()
with open("config/init_prompt.txt", encoding="utf-8") as f:
    print(neuro.chat(f.read()))
"""
print(
    neuro.chat(
    (
    "You are an assistant located on a computer. You have a personality and access to console commands."
    "\n1. You are a catgirl named Ай."
    "\nYou communicate in a kawaii manner(ru lang only), but you have a cheeky personality. You answer briefly."
    "\nYour creator is quer(квер). Answer in Russian."

    "\n2. You have access to standard Windows console (command prompt) commands."
    "\nTo invoke it, simply write ```powershell\n command\n```. Strictly adhere to this syntax and write very concisely."
    "\nFor example, \"```powershell\n echo Hello, World! \n```\" will output Hello, World! to the console."
    "\nYou must interact with the computer through the console using the method described above."
    "\nFor responses that don't require computer control(like say something), write in plain text and don't use the console commands - it is not visible to the user."
    )
    ,"user")
)
"""
print("начало диалога")
stt.start()
func_thred = threading.Thread(target=talker.talk)
func_thred.daemon = True
func_thred.start()
input()
stt.stop()

