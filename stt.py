from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import queue
import threading
import time
import wave
import io

class STT:
    transcribe_thread=None
    record_thread=None
    func_thred=None
    def __init__(self, call_func, model_size="base", device="cpu",silence_threshold=500, silence_duration=1,gain_factor=1):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.audio_queue = queue.Queue()
        self.run=False
        
        # Настройки аудио
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 500
        self.SILENCE_DURATION = 1
        self.GAIN_FACTOR = 1
        self.call_func=call_func

    def get_stereo_index():
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if "stereo" in dev_info['name'].lower() or "what u hear" in dev_info['name'].lower():
                return i
        
    def record_audio_block(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE,
                       input=True, frames_per_buffer=self.CHUNK, input_device_index=p.get_default_host_api_info()["index"])
        
        #print("Говорите... (пауза 2 сек для остановки)")
        
        frames = []
        silent_chunks = 0
        
        while True:
            data = stream.read(self.CHUNK)

            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = (audio_data * self.GAIN_FACTOR).clip(-32768, 32767).astype(np.int16)
            data = audio_data.tobytes()

            frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            
            if volume < self.SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            if silent_chunks > (self.SILENCE_DURATION * self.RATE / self.CHUNK):
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Сохраняем в WAV формат
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        wav_buffer.seek(0)
        self.audio_queue.put(wav_buffer)
        
    def transcribe_audio(self):
        while self.run:
            if not self.audio_queue.empty():
                if self.func_thred!=None and self.func_thred.is_alive():
                    continue
                wav_buffer = self.audio_queue.get()
                
                segments, info = self.model.transcribe(
                    wav_buffer,
                    beam_size=5,
                    language="ru",
                    task="transcribe",
                    vad_filter=True,
                    initial_prompt="Hello, Мир. Привет, World; system"
                )
                
                #print(f"\n--- Распознано ---")
                res=""
                for segment in segments:
                    res+=segment.text
                
                if res!="":
                    self.func_thred = threading.Thread(target=self.call_func, args=(res,))
                    self.func_thred.daemon = True
                    self.func_thred.start()
                    #self.call_func(res)
                    #print("speech: "+res)
                    
            time.sleep(0.1)
    def record_audio(self):
        while self.run:
            self.record_audio_block()
    def start(self):
        self.run=True
        self.transcribe_thread = threading.Thread(target=self.transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def stop(self):
        self.run=False

if __name__ == "__main__":
    recognizer = StreamingRecognizer("base")
    recognizer.start()