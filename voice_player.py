import queue
import threading
import sounddevice as sd
from faster_qwen3_tts import FasterQwen3TTS
import numpy as np
import time
class VoicePlayer:
    def __init__(self, target_sr=24000, blocksize=2048):
        self.queue = queue.Queue()
        self.target_sr = target_sr
        self.blocksize = blocksize
        self.stream = None
        self._finished = False
        
    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.target_sr,
            channels=1,
            dtype='float32',
            callback=self._callback,
            blocksize=self.blocksize,
            latency='low'  # Важно для минимизации задержки
        )
        self.stream.start()
        
    def _callback(self, outdata, frames, time, status):
        #if status:
            #print(f"Stream status: {status}")
            
        # Пытаемся набрать ровно 'frames' сэмплов из очереди
        current_frame = 0
        while current_frame < frames:
            try:
                # Получаем чанк из очереди (неблокирующе, но с маленьким таймаутом)
                chunk = self.queue.get(timeout=0.1)
                
                # Определяем, сколько нам нужно до конца текущего блока вывода
                needed = frames - current_frame
                available = len(chunk)
                
                if available >= needed:
                    # Чанк больше или равен нужному остатку
                    outdata[current_frame:frames] = chunk[:needed].reshape(-1, 1)
                    # Возвращаем остаток чанка обратно в начало очереди
                    if available > needed:
                        self.queue.put(chunk[needed:])
                    current_frame = frames
                else:
                    # Чанк меньше нужного, копируем его целиком
                    outdata[current_frame:current_frame + available] = chunk.reshape(-1, 1)
                    current_frame += available
                    
            except queue.Empty:
                # Если данных нет, заполняем тишиной
                # Это критично: если генерация медленная, будет тишина, но поток не упадет
                outdata[current_frame:frames] = 0
                break
                
    def add_chunk(self, audio_chunk):
        """Добавить аудио-чанк. Ожидает numpy array float32."""
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
            
        self.queue.put(audio_chunk)
        
    def wait_for_drain(self):
        """Ждать, пока очередь опустеет и все данные проиграются"""
        while not self.queue.empty():
            time.sleep(0.1)
        # Даем небольшому запасу времени на проигрывание последнего блока
        time.sleep(0.5) 
        
    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()



# Использование:
if __name__ =="__main__":
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    prompt_items = model.model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="Идет ли дождь в Калифорнии? Это единственный сон, который я видела. НЯ!!", x_vector_only_mode=True)

    print("start")
    player = VoicePlayer()
    player.start()
    for chunk, sr, _ in model.generate_voice_clone_streaming(
        text="Ваш текст здесь... И снова 3 сентября??? НЯ!!",
        language="Russian",
        voice_clone_prompt=prompt_items,
        chunk_size=4,  # меньше = меньше задержка, но больше накладных расходов
        ):
        player.add_chunk(chunk)  # чанк уже в нужном sample_rate

    player.wait_for_drain()
    player.close()