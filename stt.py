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
    def __init__(self, call_func, model_size="base", device="cpu", 
                 silence_threshold=500, silence_duration=1, 
                 gain_factor=1, use_nr=False, device_index=-1):
        """
        Initialize the Speech-to-Text engine
        :param call_func: Callback function to call when text is recognized
        :param model_size: Size of the ASR model
        :param device: Device to run the model on ('cpu' or 'cuda')
        :param silence_threshold: Threshold for detecting silence
        :param silence_duration: Duration of silence to stop recording
        :param gain_factor: Audio gain factor
        :param use_nr: Whether to use noise reduction
        :param device_index: Audio device index (-1 for default)
        """
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.audio_queue = queue.Queue()
        self.run = False
        self.use_nr = use_nr

        self.device_index = device_index
        p = pyaudio.PyAudio()
        if self.device_index == -1: 
            self.device_index = p.get_default_host_api_info()["index"]
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.gain_factor = gain_factor
        self.call_func = call_func
        self.func_thread=None

    def calibrate(self, duration=5):
        """
        Calibrate microphone sensitivity
        :param duration: Calibration duration in seconds
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format, 
            channels=self.channels, 
            rate=self.rate,
            input=True, 
            frames_per_buffer=self.chunk, 
            input_device_index=self.device_index
        )
        
        total_frames = int(self.rate / self.chunk * duration)
        calibration_sum = 0
        
        for i in range(total_frames):
            data = stream.read(self.chunk)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = (audio_data * self.gain_factor).clip(-32768, 32767).astype(np.int16)
            if self.use_nr: 
                audio_data = nr.reduce_noise(y=audio_data, sr=self.rate)
            
            calibration_sum += np.abs(audio_data).mean()
            
        stream.stop_stream()
        stream.close()
        
        self.silence_threshold = calibration_sum / total_frames
        self.silence_threshold *= 4
        
        print(f"Calibration threshold: {self.silence_threshold}")

    def record_audio_block(self):
        """
        Record a single block of audio until silence is detected
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format, 
            channels=self.channels, 
            rate=self.rate,
            input=True, 
            frames_per_buffer=self.chunk, 
            input_device_index=self.device_index
        )
        
        frames = []
        silent_chunks = 0
        record = False

        while self.run:
            data = stream.read(self.chunk)

            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = (audio_data * self.gain_factor).clip(-32768, 32767).astype(np.int16)
            if self.use_nr:
                audio_data = nr.reduce_noise(y=audio_data, sr=self.rate)
            volume = np.abs(audio_data).mean()

            data = audio_data.tobytes()

            frames.append(data)
            if not record:
                frames = frames[-int(self.rate / self.chunk * 1):]
            
            if volume < self.silence_threshold:
                silent_chunks += 1
            elif self.func_thread is None or not self.func_thread.is_alive():
                if not record: 
                    print("Recording...")
                record = True
                silent_chunks = 0
            
            if record and silent_chunks > (self.silence_duration * self.rate / self.chunk):
                break
                
        print("Recorded")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to WAV format
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        wav_buffer.seek(0)
        self.audio_queue.put(wav_buffer)
        
    def transcribe_audio(self):
        """
        Transcribe audio from the queue
        """
        while self.run:
            if not self.audio_queue.empty():
                wav_buffer = self.audio_queue.get()
                if self.func_thread is not None and self.func_thread.is_alive():
                    continue

                segments, info = self.model.transcribe(
                    wav_buffer,
                    beam_size=5,
                    language="ru",
                    task="transcribe",
                    vad_filter=True,
                    initial_prompt="Hello, Мир. Привет, World; system"
                )
                
                res = ""
                for segment in segments:
                    res += segment.text
                
                if res != "":
                    self.func_thread = threading.Thread(target=self.call_func, args=(res,))
                    self.func_thread.daemon = True
                    self.func_thread.start()
                    
            time.sleep(0.1)
    
    def record_audio(self):
        """
        Continuously record audio
        """
        while self.run:
            self.record_audio_block()
    
    def start(self):
        """
        Start the STT engine
        """
        self.run = True
        self.transcribe_thread = threading.Thread(target=self.transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def stop(self):
        """
        Stop the STT engine
        """
        self.run = False