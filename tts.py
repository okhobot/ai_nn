import torch
import sounddevice as sd
import librosa
import numpy as np
import threading
import time 
from voice_player import VoicePlayer
from faster_qwen3_tts import FasterQwen3TTS

class TTS:
    def __init__(self, speaker="ref.wav", model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        self.player = VoicePlayer()
        self.speaker = speaker
        self.play_thread = None
        # Загрузка модели
        self.model = FasterQwen3TTS.from_pretrained(model_name) 
        self.prompt_items = self.model.model.create_voice_clone_prompt(ref_audio=speaker, ref_text="Идет ли дождь в Калифорнии? Это единственный сон, который я видела. НЯ!!", x_vector_only_mode=True)

    def speak(self, text):
        self.can_play=True
        self.player.start()
        for chunk, sr, _ in self.model.generate_voice_clone_streaming(
            text=text,
            language="Russian",
            voice_clone_prompt=self.prompt_items,
            chunk_size=4,  # меньше = меньше задержка, но больше накладных расходов
            ):
            if(self.can_play):
                self.player.add_chunk(chunk)  # чанк уже в нужном sample_rate
            else:
                break
        self.player.wait_for_drain()
        self.player.close()
    
    def speak_async(self, text):
        self.stop()
        
        self.play_thread = threading.Thread(target=self.speak, args=(text,))
        self.play_thread.daemon = True
        self.play_thread.start()

    def stop(self):
        self.can_play = False
        #print(self.can_play)
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.1)
        

if __name__ == "__main__":
    tts = TTS()
    tts.speak("1 2 3 4 5")