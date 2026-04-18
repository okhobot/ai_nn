import mss
from PIL import Image
import io
import base64

class ScreenCapture:
    def __init__(self, monitor=1):
        """
        monitor: 1 - основной монитор, 0 - все мониторы, 2,3... - дополнительные
        """
        self.monitor = monitor

    def capture_to_base64(self, scale_factor: int = 1) -> str:
        """Делает скриншот и возвращает строку base64 (PNG)
        
        Args:
            scale_factor: коэффициент уменьшения разрешения (1 - без изменений, 2 - в 2 раза меньше, и т.д.)
        """
        with mss.mss() as sct:
            monitors = sct.monitors
            if self.monitor >= len(monitors):
                raise IndexError(f"Монитор {self.monitor} не найден. Доступные: {len(monitors)-1}")
            
            # Захват
            screenshot = sct.grab(monitors[self.monitor])
            # Конвертация BGRA -> RGB
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Уменьшение разрешения если нужно
            if scale_factor > 1:
                new_size = (img.width // scale_factor, img.height // scale_factor)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # В base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def save_base64_image(self, base64_data: str, filename: str = "image.png") -> str:
        """Сохраняет base64 изображение в файл
        
        Args:
            base64_data: строка base64 с изображением
            filename: имя файла для сохранения (по умолчанию 'image.png')
        
        Returns:
            Путь к сохраненному файлу
        """
        # Декодирование base64
        image_data = base64.b64decode(base64_data)
        
        # Сохранение в файл
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        return filename