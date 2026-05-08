# AI-NN - AI Neural Network Assistant

AI-NN is a modular AI assistant with speech-to-text (STT), text-to-speech (TTS), neural network processing, and memory capabilities.

## Features

- Speech-to-text conversion using Whisper models
- Text-to-speech synthesis with voice customization
- Neural network-powered conversation
- Memory module for storing and retrieving contextual information
- PowerShell command execution capability
- Configurable via JSON configuration files

## Installation

To install AI-NN as a package:

```bash
pip install -e .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

### As a Standalone Application

Run the main script directly:

```bash
python ai_nn.py
```

Or using the installed console script:

```bash
ai-nn
```

### As a Module

Import and use in your own Python code:

```python
from ai_nn import Ai_NN
from config_manager import ConfigManager

# Load configuration
cm = ConfigManager()
cm.load_config()
cm.set_hf_env()

# Initialize the AI assistant
ai_nn = Ai_NN(cm.get_json_config())

# Start chatting
ai_nn.chat("Hello, how are you?")
```

## Configuration

Create a `config/config.json` file with your settings. See `config_template.json` for an example.

## Requirements

- Python 3.8+
- Hardware requirements vary based on model sizes used

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the [LICENSE.md](LICENSE.md) file for details.