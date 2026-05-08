"""
AI-NN - AI Neural Network Assistant Package
A modular AI assistant with STT, TTS, and neural network capabilities.
"""

from .ai_nn import Ai_NN
from .config_manager import ConfigManager
from .nn import NN
from .tts import TTS
from .stt import STT
from .memory_module import MemoryModule

__all__ = ['Ai_NN', 'ConfigManager', 'NN', 'TTS', 'STT', 'MemoryModule']