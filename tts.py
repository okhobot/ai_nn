import torch
import sounddevice as sd
import librosa
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
import threading
import time 

class TTS:
    speaker = "kseniya"  # aidar, baya, kseniya, eugene, xenia
    model = None
    pitch_shift = 0
    can_play = True
    
    def __init__(self, pitch_shift=0, speaker="kseniya", model="v5_1_ru"):
        model_path = "maximxls/text-normalization-ru-terrible"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.normalizer = T5ForConditionalGeneration.from_pretrained(model_path)
        
        self.speaker = speaker
        self.pitch_shift = pitch_shift
        self.play_thread = None
        # Загрузка модели
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                model='silero_tts',
                                language='ru',
                                speaker=model
                                )
    
    def _normalize_text(self, text):
        # Проверяем, есть ли что нормализовать
        if self._needs_normalization(text):
            inp_ids = self.tokenizer(text, return_tensors="pt").input_ids
            out_ids = self.normalizer.generate(inp_ids, max_new_tokens=512)[0]
            result = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            return result
        else:
            return text  # возвращаем как есть

    def _needs_normalization(self, text):
        """Проверяет, нужна ли нормализация"""
        # Проверяем наличие чисел
        if any(c.isdigit() for c in text):
            return True
        if any((c.lower() in "qwertyuiopasdfghjklzxcvbnm") for c in text):
            return True
        # Проверяем наличие дат (через регулярку)
        import re
        if re.search(r'\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}', text):
            return True
        # Проверяем наличие времени
        if re.search(r'\d{1,2}:\d{2}', text):
            return True
        # Проверяем наличие URL/email
        if '@' in text or 'http' in text:
            return True
        return False

    def generate_speech(self, text):
        # Генерация речи
        self.can_play = True
        #print(self._normalize_text(text))
        audio = self.model.apply_tts(text=self._normalize_text(text),
                                speaker=self.speaker,
                                sample_rate=48000)
        audio_np = audio.numpy()
        audio_shifted = librosa.effects.pitch_shift(audio_np, sr=48000, n_steps=self.pitch_shift)
        return audio_shifted

    def speak(self, text):
        sd.play(self.generate_speech(text), 48000)
        while self.can_play and sd.get_stream().active:
            #print(self.can_paly)
            time.sleep(0.01)
    
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
    tts = TTS(3, "kseniya", "v5_1_ru")
    #tts.speak("Привет, мир!")
    """
    text="привет, мир. И снова третье сентября..."
    speech=[]
    for word in [text]:
        print(word)
        speech.append(tts.generate_speech(word))

    sd.play(speech[0],48000)
    speech.pop(0)

    while True:
        if  not sd.get_stream().active and len(speech)>0:
            sd.play(speech[0],48000)
            speech.pop(0)
            """

    tts.speak_async("привет, мир! И снова 3 сентября...")
    #time.sleep(5)
    #tts.stop()
    #tts.speak_async("Translation: To go to Office, press the  button and search for in the search bar.")
    input()