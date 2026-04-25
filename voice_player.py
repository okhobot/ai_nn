import queue
import threading
import sounddevice as sd
from faster_qwen3_tts import FasterQwen3TTS
import numpy as np
import time

class VoicePlayer:
    def __init__(self, target_sr=24000, blocksize=2048):
        self.target_sr = target_sr
        self.blocksize = blocksize
        self.stream = None
        self._reset_state()  # Инициализируем чистое состояние

    def _reset_state(self):
        """Полный сброс внутренних переменных для безопасного перезапуска"""
        self.queue = queue.Queue()
        self._pending_chunk = None
        self._finished_adding = False
        self._drain_event = threading.Event()

    def start(self):
        # Если поток уже существует, корректно останавливаем и закрываем его
        if self.stream is not None:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None

        self._reset_state()  # ← Ключевое: очищаем очередь, флаги и события

        self.stream = sd.OutputStream(
            samplerate=self.target_sr,
            channels=1,
            dtype='float32',
            callback=self._callback,
            blocksize=self.blocksize,
            latency='low'
        )
        self.stream.start()

    def _callback(self, outdata, frames, time, status):
        if status:
            print(f"⚠️ Audio stream status: {status}")

        idx = 0
        while idx < frames:
            # 1. Берём остаток от прошлого вызова
            if self._pending_chunk is not None:
                chunk = self._pending_chunk
                self._pending_chunk = None
            else:
                # 2. Берём новый чанк без блокировки
                try:
                    chunk = self.queue.get_nowait()
                except queue.Empty:
                    if self._finished_adding:
                        self._drain_event.set()  # Сигнал главному потоку
                    outdata[idx:frames, 0] = 0.0
                    return

            needed = frames - idx
            available = len(chunk)

            if available >= needed:
                outdata[idx:frames, 0] = chunk[:needed]
                if available > needed:
                    self._pending_chunk = chunk[needed:]  # Сохраняем хвост
                idx = frames
            else:
                outdata[idx:idx + available, 0] = chunk
                idx += available

    def add_chunk(self, audio_chunk):
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        self.queue.put(audio_chunk)

    def stop(self):
        """Немедленно прервать воспроизведение и разблокировать wait_for_drain"""
        if self.stream and self.stream.active:
            self.stream.stop()
        self._finished_adding = True
        self._drain_event.set()

    def wait_for_drain(self):
        """Блокирует выполнение, пока не проиграются все данные"""
        self._finished_adding = True
        self._drain_event.wait()
        time.sleep(0.05)  # Буферная защита от последнего блока

    def close(self):
        if self.stream:
            self.stop()
            self.stream.close()
            self.stream = None


# Использование:
if __name__ =="__main__":
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    prompt_items = model.model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="Идет ли дождь в Калифорнии? Это единственный сон, который я видела. НЯ!!", x_vector_only_mode=True)

    print("start")
    player = VoicePlayer()
    player.start()
    for chunk, sr, _ in model.generate_voice_clone_streaming(
        text="Идет ли дождь в Калифорнии? Это единственный сон, который я видела. НЯ!!",
        language="Russian",
        voice_clone_prompt=prompt_items,
        chunk_size=4,  # меньше = меньше задержка, но больше накладных расходов
        ):
        player.add_chunk(chunk)  # чанк уже в нужном sample_rate

    player.wait_for_drain()
    player.close()