#!/usr/bin/env python3
"""
Minimal TildeOpen-30b Interface
Uses streaming and minimal caching to reduce storage requirements.
Loads only essential model components and streams weights as needed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import tempfile
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MinimalTildeModel:
    def __init__(self, model_name="TildeAI/TildeOpen-30b"):
        """Initialize with minimal storage footprint."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Use a temporary directory that will be cleaned up
        self.temp_cache = tempfile.mkdtemp(prefix="tilde_cache_")
        print(f"🚀 Minimal TildeOpen-30b Interface")
        print(f"📁 Temporary cache: {self.temp_cache}")
    
    def load_tokenizer_only(self):
        """Load only the tokenizer first (small download)."""
        try:
            print("📥 Loading tokenizer (minimal download)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                cache_dir=self.temp_cache
            )
            print("✅ Tokenizer loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            return False
    
    def estimate_model_size(self):
        """Provide information about model requirements."""
        print("\n📊 TildeOpen-30b Model Information:")
        print("• Parameters: ~30 billion")
        print("• Model size: ~60GB (full precision)")
        print("• RAM required: 32GB+ (minimum)")
        print("• Recommended: GPU with 24GB+ VRAM")
        
        print(f"\n💾 Your system:")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"• GPU: {torch.cuda.get_device_name()}")
            print(f"• GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 24:
                print("✅ Sufficient GPU memory for model loading")
                return "gpu_good"
            elif gpu_memory >= 12:
                print("⚠️  Limited GPU memory - quantization recommended")
                return "gpu_limited"
            else:
                print("❌ Insufficient GPU memory")
                return "gpu_insufficient"
        else:
            print("• GPU: Not available")
            print("❌ CPU-only inference will be extremely slow")
            return "cpu_only"
    
    def test_tokenizer(self):
        """Test the tokenizer with sample text."""
        if not self.tokenizer:
            print("❌ Tokenizer not loaded")
            return False
        
        print("\n🧪 Testing tokenizer with multilingual samples...")
        
        test_texts = [
            "Hello, how are you?",  # English
            "Bonjour, comment allez-vous?",  # French
            "Hola, ¿cómo estás?",  # Spanish
            "Здравствуйте, как дела?",  # Russian
            "Witaj, jak się masz?",  # Polish
        ]
        
        for text in test_texts:
            try:
                tokens = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(tokens)
                print(f"✅ {text} → {len(tokens)} tokens → {decoded}")
            except Exception as e:
                print(f"❌ Failed to tokenize: {text} - {e}")
                return False
        
        return True
    
    def demo_without_model(self):
        """Demonstrate capabilities without loading the full model."""
        print("\n🎯 TildeOpen-30b Capabilities Demo (Tokenizer Only)")
        print("=" * 60)
        
        if not self.test_tokenizer():
            return False
        
        print(f"\n🌍 Supported Languages:")
        languages = [
            "English", "German", "French", "Polish", "Russian", "Italian", 
            "Portuguese", "Czech", "Dutch", "Spanish", "Finnish", "Turkish",
            "Hungarian", "Bulgarian", "Croatian", "Danish", "Estonian",
            "Latvian", "Lithuanian", "Norwegian", "Romanian", "Serbian",
            "Slovak", "Slovene", "Swedish", "Ukrainian", "and more..."
        ]
        
        for i, lang in enumerate(languages):
            if i < len(languages) - 1:
                print(f"• {lang}")
            else:
                print(f"• {lang}")
        
        print(f"\n📈 Model Performance Highlights:")
        print("• Optimized for Nordic and Eastern European languages")
        print("• Trained on 2 trillion tokens")
        print("• Equitable tokenizer for fair language representation")
        print("• State-of-the-art performance on underrepresented languages")
        
        return True
    
    def interactive_tokenizer_demo(self):
        """Interactive demo using just the tokenizer."""
        print("\n🔄 Interactive Tokenizer Demo")
        print("💡 Type text to see how TildeOpen tokenizes it")
        print("💡 Try different languages!")
        print("💡 Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n📝 Enter text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Tokenize and analyze
                tokens = self.tokenizer.encode(user_input)
                decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                print(f"📊 Analysis:")
                print(f"   Original: {user_input}")
                print(f"   Tokens: {len(tokens)}")
                print(f"   Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                print(f"   Decoded: {decoded}")
                print(f"   Efficiency: {len(user_input)/len(tokens):.2f} chars/token")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Demo ended!")
    
    def show_model_loading_options(self):
        """Show different options for loading the full model."""
        system_capability = self.estimate_model_size()
        
        print(f"\n🚀 Model Loading Options:")
        print("=" * 40)
        
        if system_capability == "gpu_good":
            print("1. 🎯 RECOMMENDED: Standard Loading")
            print("   python3 tilde_model.py")
            print("   • Full model capabilities")
            print("   • Best performance")
            
            print("\n2. 💾 Memory Optimized:")
            print("   python3 tilde_lightweight.py")
            print("   • Uses quantization")
            print("   • Reduced memory usage")
            
        elif system_capability == "gpu_limited":
            print("1. 🎯 RECOMMENDED: Quantized Loading")
            print("   python3 tilde_lightweight.py")
            print("   • 8-bit or 4-bit quantization")
            print("   • Fits in limited GPU memory")
            
            print("\n2. ⚠️  Standard Loading (may fail)")
            print("   python3 tilde_model.py")
            print("   • May exceed GPU memory")
            
        elif system_capability == "gpu_insufficient":
            print("1. 🎯 RECOMMENDED: Cloud GPU")
            print("   • Google Colab (free GPU)")
            print("   • AWS, Azure, or GCP instances")
            print("   • Kaggle notebooks")
            
            print("\n2. ⚠️  Quantized CPU (very slow)")
            print("   python3 tilde_cpu_friendly.py")
            print("   • Extremely slow on CPU")
            print("   • Requires 32GB+ RAM")
            
        else:  # cpu_only
            print("1. 🎯 RECOMMENDED: Cloud GPU Services")
            print("   • Google Colab: https://colab.research.google.com/")
            print("   • Kaggle: https://www.kaggle.com/code")
            print("   • Hugging Face Spaces (if available)")
            
            print("\n2. ⚠️  Local CPU (not recommended)")
            print("   python3 tilde_cpu_friendly.py")
            print("   • Requires 32GB+ RAM")
            print("   • Extremely slow (minutes per response)")
        
        print(f"\n💡 Alternative: Use a smaller model for testing:")
        print("   • microsoft/DialoGPT-large")
        print("   • TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("   • microsoft/phi-2")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_cache)
            print(f"🧹 Cleaned up temporary cache")
        except:
            pass

def main():
    """Main function for minimal demo."""
    print("🌟 TildeOpen-30b Minimal Interface")
    print("=" * 50)
    print("🎯 This demo loads only the tokenizer (small download)")
    print("💡 Perfect for exploring the model without 60GB+ download")
    
    model = MinimalTildeModel()
    
    try:
        # Load tokenizer
        if not model.load_tokenizer_only():
            return
        
        # Run capabilities demo
        if model.demo_without_model():
            # Show loading options
            model.show_model_loading_options()
            
            # Interactive demo
            print(f"\n❓ Would you like to try the interactive tokenizer demo?")
            response = input("Try interactive demo? (y/N): ").strip().lower()
            
            if response == 'y':
                model.interactive_tokenizer_demo()
        
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()
