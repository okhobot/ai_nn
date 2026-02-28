import torch
import sounddevice as sd
import librosa
from runorm import RUNorm

import re

class TTS:
    speaker="kseniya"#aidar, baya, kseniya, eugene, xenia
    model=None
    normalizer=None
    pithc_shift=0
    def __init__(self, pithc_shift=0, speaker="kseniya", model="v5_1_ru"):
        self.normalizer = RUNorm()
        self.normalizer.load(model_size="small", device="cpu")
        self.speaker=speaker
        self.pithc_shift=pithc_shift
        # Загрузка модели
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                model='silero_tts',
                                language='ru',
                                speaker=model)
    


    def speak(self, text):
        # Генерация речи
        audio = self.model.apply_tts(text=self.normalizer.norm(text),
                                speaker=self.speaker,
                                sample_rate=48000)
        audio_np = audio.numpy()
        audio_shifted = librosa.effects.pitch_shift(audio_np, sr=48000, n_steps=2)
        sd.play(audio_shifted, 48000)
        sd.wait()
        
    def stop(self):
        sd.stop()

if __name__=="__main__":
    tts=TTS(4,"kseniya","v5_1_ru")
    tts.speak("привет, мир! И снова 3 сентября...")
    tts.speak("1 2 3 4 5 6 7 8 9 10. system32 call")