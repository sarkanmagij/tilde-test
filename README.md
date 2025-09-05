# TildeOpen-30b Model Integration

A Python project for connecting to and interacting with the [TildeAI/TildeOpen-30b](https://huggingface.co/TildeAI/TildeOpen-30b) language model via Hugging Face transformers.

## About TildeOpen-30b

TildeOpen-30b is a 30B parameter open-source foundational language model built to serve underrepresented Nordic and Eastern European languages. It supports **34 languages** including English, German, French, Polish, Russian, and many others, representing over 165 million speakers.

### Key Features
- 🌍 **Multilingual**: 34 European languages supported
- 🔓 **Open Source**: CC-BY-4.0 license
- ⚡ **Optimized**: Equitable tokenizer for fair language representation
- 🎯 **Focused**: Specialized for Nordic and Eastern European languages
- 🏗️ **Enterprise Ready**: Trained on LUMI supercomputer with 768 AMD MI250X GPUs

## Quick Start

### 1. Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd tilde-test

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Test Connection

```bash
# Quick connection test
python3 test_connection.py
```

### 3. Interactive Chat

```bash
# Start interactive chat with the model
python3 tilde_model.py
```

## Project Structure

```
tilde-test/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── tilde_model.py              # Main model interface with chat functionality
├── test_connection.py          # Quick connection test script
└── TildeOpen-30b-README.md     # Official model documentation
```

## Dependencies

- **torch** (≥2.0.0) - PyTorch framework
- **transformers** (≥4.35.0) - Hugging Face transformers
- **accelerate** (≥0.20.0) - Model acceleration
- **safetensors** (≥0.3.0) - Safe tensor serialization
- **sentencepiece** (≥0.1.99) - Tokenization (required for this model)
- **protobuf** (≥3.20.0) - Protocol buffers

## Usage Examples

### Basic Text Generation

```python
from tilde_model import TildeOpenModel

# Initialize and load model
model = TildeOpenModel()
model.load_model()

# Generate text
response = model.generate_text("Hello, how are you today?", max_new_tokens=100)
print(response)
```

### Interactive Chat

The `tilde_model.py` script provides an interactive chat interface:

```bash
python3 tilde_model.py
```

Features:
- 💬 Interactive conversation
- 🧹 Clear conversation history with `clear`
- 🚪 Exit with `quit`, `exit`, or `q`
- 🎛️ Configurable generation parameters

### Direct Model Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer (use_fast=False is required!)
tokenizer = AutoTokenizer.from_pretrained("TildeAI/TildeOpen-30b", use_fast=False)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "TildeAI/TildeOpen-30b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## System Requirements

### Minimum Requirements
- **RAM**: 32GB+ recommended (for 30B parameter model)
- **GPU**: 24GB+ VRAM recommended (RTX 3090, RTX 4090, A100, etc.)
- **Storage**: 65GB+ free space for model files
- **Python**: 3.8+

### Performance Notes
- 🖥️ **CPU Only**: Will work but be extremely slow
- 🎮 **GPU Recommended**: Significant performance improvement
- ☁️ **Cloud Options**: Consider Google Colab Pro, AWS, or similar for GPU access

## Supported Languages

Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hungarian, Icelandic, Irish, Italian, Latvian, Lithuanian, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovene, Spanish, Swedish, Turkish, Ukrainian, and more.

## Model Performance

TildeOpen-30b shows excellent performance on underrepresented European languages compared to other large language models. See [TildeOpen-30b-README.md](./TildeOpen-30b-README.md) for detailed benchmarks.

## Troubleshooting

### Common Issues

1. **SentencePiece Error**: Install with `pip install sentencepiece`
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Slow Generation**: Ensure GPU is being used properly
4. **Model Download Fails**: Check internet connection and disk space

### GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source. The TildeOpen-30b model is licensed under CC-BY-4.0.

## Links

- 🤗 [Model on Hugging Face](https://huggingface.co/TildeAI/TildeOpen-30b)
- 🏢 [Tilde.ai](https://tilde.ai/tildeopen-llm/)
- 📊 [TILDE Bench](https://tilde-nlp.github.io/tokenizer-bench.html)
- 🇪🇺 [EuroHPC JU Large AI Grand Challenge](https://www.eurohpc-ju.europa.eu/winners-announced-large-ai-grand-challenge-2024-06-26_en)

---

**Note**: This is a foundational model not yet adapted for instruction following or safety alignment. The next version will be a specialized translation model.