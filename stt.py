from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import queue
import threading
import time
import wave
import io
import noisereduce as nr


class STT:
    transcribe_thread=None
    record_thread=None
    func_thred=None
    def __init__(self, call_func, model_size="base", device="cpu",silence_threshold=500, silence_duration=1,gain_factor=1, use_nr=False, device_index=-1):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.audio_queue = queue.Queue()
        self.run=False
        self.use_nr=use_nr

        self.device_index=device_index
        p = pyaudio.PyAudio()
        if device_index==-1: self.device_index=p.get_default_host_api_info()["index"]
        
        # Настройки аудио
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 500
        self.SILENCE_DURATION = 1
        self.GAIN_FACTOR = 1
        self.call_func=call_func

            
            
    def calibrate(self, time=5):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE,
                    input=True, frames_per_buffer=self.CHUNK, 
                    input_device_index=self.device_index)
        
        frames = []
        # Рассчитываем, сколько фрагментов нужно для 5 секунд записи
        total_frames = int(self.RATE / self.CHUNK * time)
        
        for i in range(total_frames):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Объединяем все фрагменты в один байтовый поток
        full_data = b''.join(frames)
        
        # Преобразуем в numpy массив для обработки
        audio_data = np.frombuffer(full_data, dtype=np.int16)
        audio_data = (audio_data * self.GAIN_FACTOR).clip(-32768, 32767).astype(np.int16)
        if self.use_nr: 
            audio_data = nr.reduce_noise(y=audio_data, sr=self.RATE)
        
        self.SILENCE_THRESHOLD = (np.abs(audio_data).max()*2+np.abs(audio_data).mean())/3
        print("st: ", self.SILENCE_THRESHOLD, len(audio_data))

        
    def record_audio_block(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE,
                       input=True, frames_per_buffer=self.CHUNK, input_device_index=self.device_index)
        
        #print("Говорите... (пауза 2 сек для остановки)")
        
        frames = []
        silent_chunks = 0
        record=False

        while True:
            data = stream.read(self.CHUNK)

            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = (audio_data * self.GAIN_FACTOR).clip(-32768, 32767).astype(np.int16)
            if self.use_nr:
                audio_data = nr.reduce_noise(y=audio_data, sr=self.RATE)
            volume = np.abs(audio_data).mean()

            data = audio_data.tobytes()

            frames.append(data)
            if not record:
                frames=frames[-int(self.RATE / self.CHUNK * 1):]
            
            #print(volume)
            
            if volume < self.SILENCE_THRESHOLD:
                silent_chunks += 1
            elif self.func_thred==None or not self.func_thred.is_alive():
                    silent_chunks = 0
                    if not record: print("recording")
                    record=True
                    silent_chunks=0
            
            if record and silent_chunks > (self.SILENCE_DURATION * self.RATE / self.CHUNK):
                break
        print("recorded")
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

        #with open("test.wav", 'wb') as f: f.write(wav_buffer.getvalue())
        
        wav_buffer.seek(0)
        self.audio_queue.put(wav_buffer)
        
    def transcribe_audio(self):
        while self.run:
            if not self.audio_queue.empty():
                wav_buffer = self.audio_queue.get()
                if self.func_thred!=None and self.func_thred.is_alive():
                    continue

                
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
    recognizer = STT(print, "base", use_nr=False, device_index=0)
    
    recognizer.calibrate(2)
    print("start")
    recognizer.start()
    input()
    recognizer.stop()