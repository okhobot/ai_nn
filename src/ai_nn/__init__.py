"""
AI-NN - AI Neural Network Assistant Package
A modular AI assistant with STT, TTS, and neural network capabilities.
"""

try:
    # Импорт как относительные (при использовании как пакет)
    from .ai_nn import Ai_NN, Talker, main
    from .config_manager import ConfigManager
    from .nn import NN
    from .tts import TTS
    from .stt import STT
    from .memory_module import MemoryModule
except ImportError:
    # Импорт как абсолютные (при прямом запуске)
    from ai_nn import Ai_NN, Talker, main
    from config_manager import ConfigManager
    from nn import NN
    from tts import TTS
    from stt import STT
    from memory_module import MemoryModule