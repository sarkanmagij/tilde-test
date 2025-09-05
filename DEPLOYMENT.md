# 🚀 TildeOpen-30b Cloud Deployment Guide

This guide shows you how to deploy TildeOpen-30b on various cloud platforms, perfect for M2 Mac users with limited local resources.

## 📓 Google Colab (Free GPU)

### Option 1: Jupyter Notebook (Recommended)

1. **Open the Colab Notebook:**
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/tilde-test/blob/main/TildeOpen_30b_Colab.ipynb)

2. **Enable GPU:**
   - Go to `Runtime` > `Change runtime type`
   - Select `Hardware accelerator: GPU`
   - Choose `GPU type: T4` (or better if available)

3. **Run the notebook:**
   - Execute all cells in order
   - The notebook handles everything automatically

### Option 2: Python Script

1. **Upload script to Colab:**
   ```python
   # Upload tilde_colab_optimized.py to your Colab session
   !wget https://raw.githubusercontent.com/YOUR_USERNAME/tilde-test/main/tilde_colab_optimized.py
   ```

2. **Run the script:**
   ```python
   !python3 tilde_colab_optimized.py
   ```

### Colab Pro Benefits
- Longer runtimes (up to 24 hours)
- Priority access to GPUs
- Background execution
- More memory (High-RAM option)

## 🤗 Hugging Face Spaces

### Setup Instructions

1. **Create a new Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the SDK

2. **Configure your Space:**
   ```yaml
   # Copy settings from README_spaces.md
   title: TildeOpen-30b Chat
   emoji: 🌍
   sdk: gradio
   sdk_version: 4.0.0
   app_file: app.py
   ```

3. **Upload files:**
   - `app.py` (main Gradio interface)
   - `requirements_spaces.txt` → rename to `requirements.txt`
   - `README_spaces.md` → rename to `README.md`

4. **Set hardware:**
   - Go to Space settings
   - Select "GPU" hardware
   - Choose appropriate GPU tier

5. **Deploy:**
   - Your Space will build automatically
   - Share the URL with others!

### Space Features
- **Web interface** with chat functionality
- **Multilingual support** for 34 languages
- **Adjustable parameters** (temperature, max tokens)
- **Public or private** access
- **Custom domain** support (Pro)

## ☁️ Other Cloud Options

### Kaggle Notebooks
1. Go to [Kaggle](https://www.kaggle.com/code)
2. Create new notebook
3. Enable GPU in settings
4. Upload and run your code

### AWS SageMaker
1. Create SageMaker notebook instance
2. Choose GPU-enabled instance type
3. Upload your code and run

### Google Cloud Platform
1. Create Compute Engine instance with GPU
2. Install dependencies
3. Run your scripts

### Azure Machine Learning
1. Create compute instance
2. Enable GPU
3. Upload and execute notebooks

## 📱 Mobile Access

### Progressive Web App (PWA)
Your Hugging Face Space can be used as a mobile app:
1. Open your Space URL in mobile browser
2. Add to home screen
3. Use like a native app

### Responsive Design
The Gradio interface is mobile-friendly and works well on:
- iPhones
- Android devices  
- Tablets
- Desktop browsers

## 🔧 Performance Optimization

### GPU Memory Management
```python
# Automatic memory cleanup in our implementations
torch.cuda.empty_cache()
gc.collect()
```

### Quantization Options
- **4-bit**: Maximum memory savings, slight quality loss
- **8-bit**: Balanced performance and memory
- **16-bit**: Best quality, more memory usage

### Batch Processing
For multiple requests:
```python
# Process requests in batches to optimize GPU usage
batch_size = 4
```

## 🚨 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Use 4-bit quantization
- Reduce max_new_tokens
- Clear cache between generations

**"Model loading failed"**
- Check internet connection
- Verify GPU is enabled
- Try restarting runtime

**"Slow responses"**
- Ensure GPU is being used
- Check if model is quantized
- Reduce context length

### Debug Commands
```python
# Check GPU status
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name()}")

# Memory usage
memory_used = torch.cuda.memory_allocated() / 1024**3
print(f"GPU memory used: {memory_used:.1f}GB")
```

## 💰 Cost Comparison

| Platform | Free Tier | Paid Options | Best For |
|----------|-----------|--------------|----------|
| Google Colab | 12hr sessions, T4 GPU | Pro: $10/month | Development & Testing |
| HF Spaces | CPU only | GPU: $0.60/hour | Production Apps |
| Kaggle | 30hr/week GPU | None | Competitions & Learning |
| AWS | Free tier limited | Pay per use | Enterprise |
| GCP | $300 credit | Pay per use | Scalable Apps |

## 🔐 Security & Privacy

### Data Privacy
- Inputs are processed on cloud servers
- Check platform privacy policies
- Consider data sensitivity

### API Keys
- Use environment variables for secrets
- Don't commit API keys to repositories
- Rotate keys regularly

### Access Control
- Set appropriate Space visibility
- Use authentication when needed
- Monitor usage logs

---

## 🎯 Recommendations for M2 Mac Users

1. **Start with:** Google Colab notebook (free, powerful)
2. **For production:** Hugging Face Spaces with GPU
3. **For experimentation:** Kaggle notebooks
4. **For enterprise:** AWS/GCP with proper scaling

**Happy deploying! 🚀**
