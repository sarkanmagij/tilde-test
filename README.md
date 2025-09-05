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
git clone <https://github.com/sarkanmagij/tilde-test>
cd tilde-test

# Install dependencies
pip3 install -r requirements.txt
```

### 2. Choose Your Approach

#### 📓 **Google Colab** (Recommended for M2 Mac with 8GB RAM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/tilde-test/blob/main/TildeOpen_30b_Colab.ipynb)
- ✅ Free GPU access
- ✅ No local storage required
- ✅ Full model capabilities
- ✅ Interactive chat interface

#### 📡 **API Demo** (Zero Setup Required)
```bash
# Test immediately without any downloads
python3 tilde_api_demo.py
```
- ✅ Uses Hugging Face Inference API
- ✅ No model downloads
- ✅ Works on any system
- ✅ Immediate testing

#### 🚀 **Minimal Demo** (Local Exploration)
```bash
# Explore the model without downloading 60GB+ files
python3 tilde_minimal.py
```
- ✅ Only downloads tokenizer (~2.3MB)
- ✅ Works on any system
- ✅ Demonstrates multilingual capabilities
- ✅ Shows system requirements for full model

#### 🎮 **GPU Users** (24GB+ VRAM)
```bash
# Full model with optimizations
python3 tilde_lightweight.py
```
- ✅ 8-bit/4-bit quantization
- ✅ Automatic memory management
- ✅ Best performance/memory balance

#### 💻 **CPU Users** (32GB+ RAM)
```bash
# CPU-optimized version (very slow)
python3 tilde_cpu_friendly.py
```
- ⚠️ Extremely slow on CPU
- ⚠️ Requires significant RAM
- 💡 Cloud GPU recommended instead

#### 🔧 **Standard Approach** (48GB+ GPU)
```bash
# Original full model loading
python3 tilde_model.py
```

#### 🧪 **Quick Test** (Downloads full model)
```bash
# Test connection with full model download
python3 test_connection.py
```

## Project Structure

```
tilde-test/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── tilde_minimal.py            # 🚀 Minimal demo (tokenizer only, ~2MB download)
├── tilde_lightweight.py        # 🎮 GPU optimized with quantization
├── tilde_cpu_friendly.py       # 💻 CPU optimized (slow, high RAM)
├── tilde_model.py              # 🔧 Standard model interface
├── test_connection.py          # 🧪 Full model connection test
├── tilde_api_demo.py           # 📡 API-based demo (zero downloads!)
├── TildeOpen_30b_Colab.ipynb   # 📓 Google Colab notebook
├── tilde_colab_optimized.py    # 🚀 Colab-optimized script
├── app.py                      # 🌐 Gradio web interface (HF Spaces)
├── requirements_spaces.txt     # 📦 Dependencies for HF Spaces
└── TildeOpen-30b-README.md     # Official model documentation
```

## Dependencies

- **torch** (≥2.0.0) - PyTorch framework
- **transformers** (≥4.35.0) - Hugging Face transformers
- **accelerate** (≥0.20.0) - Model acceleration
- **safetensors** (≥0.3.0) - Safe tensor serialization
- **sentencepiece** (≥0.1.99) - Tokenization (required for this model)
- **protobuf** (≥3.20.0) - Protocol buffers

## Lightweight Options (No 60GB Download!)

### 🚀 Minimal Demo - Start Here!
Perfect for exploring TildeOpen-30b without massive downloads:

```bash
python3 tilde_minimal.py
```

**What it does:**
- Downloads only the tokenizer (~2.3MB)
- Demonstrates multilingual tokenization
- Shows system requirements
- Interactive tokenizer testing
- Zero model weight downloads

**Best for:**
- First-time exploration
- Understanding model capabilities
- Testing tokenization across 34 languages
- Checking system compatibility

### 💾 Memory-Optimized Loading
For users who want the full model with minimal memory usage:

```bash
python3 tilde_lightweight.py
```

**Features:**
- Automatic quantization (4-bit/8-bit)
- Streams model weights as needed
- Adaptive memory management
- GPU/CPU detection
- Reduced storage requirements

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
- ☁️ **Cloud Options**: Google Colab, Hugging Face Spaces, AWS, or similar for GPU access

## 🚀 Cloud Deployment Options

### 📓 Google Colab (Recommended for M2 Mac users)

**Perfect for your M2 Mac with 8GB RAM!**

1. **Open the Colab Notebook:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/tilde-test/blob/main/TildeOpen_30b_Colab.ipynb)

2. **Enable GPU:** Go to `Runtime` > `Change runtime type` > Select `GPU`

3. **Run all cells** in order - the notebook includes:
   - Automatic dependency installation
   - Model loading with quantization
   - Interactive chat interface
   - Multilingual testing examples

4. **Alternative Colab Script:**
   ```bash
   # Upload tilde_colab_optimized.py to Colab and run:
   python3 tilde_colab_optimized.py
   ```

### 🤗 Hugging Face Spaces

**Deploy your own web interface:**

1. **Fork this repository**
2. **Create a new Space** on Hugging Face
3. **Upload these files:**
   - `app.py` (main Gradio interface)
   - `requirements_spaces.txt` (rename to `requirements.txt`)
   - `README_spaces.md` (rename to `README.md`)
4. **Set GPU hardware** in Space settings
5. **Your web interface will be live!**

### ⚡ Quick Cloud Start

**For immediate testing without setup:**

```python
# Use the API demo (no local downloads)
python3 tilde_api_demo.py
```

This uses Hugging Face's Inference API - zero setup required!

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