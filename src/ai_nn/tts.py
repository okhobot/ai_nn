import torch
import sounddevice as sd
import librosa
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
import threading
import time
import re


class TTS:
    def __init__(self, pitch_shift=0, speaker="kseniya", model="v5_1_ru"):
        """
        Initialize the Text-to-Speech engine
        :param pitch_shift: Pitch shift amount
        :param speaker: Speaker name
        :param model: Model name
        """
        model_path = "maximxls/text-normalization-ru-terrible"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.normalizer = T5ForConditionalGeneration.from_pretrained(model_path)
        
        self.speaker = speaker
        self.pitch_shift = pitch_shift
        self.play_thread = None
        self.can_play = True
        
        # Load model
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker=model
        )
    
    def _normalize_text(self, text):
        """
        Normalize text before speech synthesis
        :param text: Input text
        :return: Normalized text
        """
        if self._needs_normalization(text):
            inp_ids = self.tokenizer(text, return_tensors="pt").input_ids
            out_ids = self.normalizer.generate(inp_ids, max_new_tokens=512)[0]
            result = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            return result
        else:
            return text  # return as is

    def _needs_normalization(self, text):
        """
        Check if text needs normalization
        :param text: Input text
        :return: True if normalization is needed
        """
        # Check for digits
        if any(c.isdigit() for c in text):
            return True
        # Check for Latin characters
        if any((c.lower() in "qwertyuiopasdfghjklzxcvbnm") for c in text):
            return True
        # Check for dates (via regex)
        if re.search(r'\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}', text):
            return True
        # Check for time
        if re.search(r'\d{1,2}:\d{2}', text):
            return True
        # Check for URL/email
        if '@' in text or 'http' in text:
            return True
        return False

    def generate_speech(self, text):
        """
        Generate speech from text
        :param text: Input text
        :return: Audio array
        """
        self.can_play = True
        audio = self.model.apply_tts(
            text=self._normalize_text(text),
            speaker=self.speaker,
            sample_rate=48000
        )
        audio_np = audio.numpy()
        audio_shifted = librosa.effects.pitch_shift(audio_np, sr=48000, n_steps=self.pitch_shift)
        return audio_shifted

    def speak(self, text):
        """
        Speak text synchronously
        :param text: Text to speak
        """
        sd.play(self.generate_speech(text), 48000)
        while self.can_play and sd.get_stream().active:
            time.sleep(0.01)
    
    def speak_async(self, text):
        """
        Speak text asynchronously
        :param text: Text to speak
        """
        self.stop()
        
        self.play_thread = threading.Thread(target=self.speak, args=(text,))
        self.play_thread.daemon = True
        self.play_thread.start()

    def stop(self):
        """
        Stop current speech playback
        """
        self.can_play = False
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.1)