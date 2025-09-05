#!/usr/bin/env python3
"""
TildeOpen-30b Colab Optimized Script
Streamlined version specifically designed for Google Colab environments.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import gc
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class ColabTildeModel:
    """Optimized TildeOpen-30b model for Google Colab."""
    
    def __init__(self):
        self.model_name = "TildeAI/TildeOpen-30b"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        print(f"🚀 TildeOpen-30b Colab Interface")
        print(f"📱 Device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        
    def check_colab_environment(self):
        """Check if running in Google Colab."""
        try:
            import google.colab
            print("✅ Running in Google Colab")
            return True
        except ImportError:
            print("⚠️ Not running in Google Colab - some features may not work")
            return False
    
    def install_dependencies(self):
        """Install required dependencies in Colab."""
        packages = [
            "transformers>=4.35.0",
            "accelerate>=0.20.0", 
            "bitsandbytes>=0.41.0",
            "sentencepiece>=0.1.99"
        ]
        
        print("🔧 Installing dependencies...")
        for package in packages:
            os.system(f"pip install -q {package}")
        print("✅ Dependencies installed!")
    
    def load_model_colab(self, use_quantization=True):
        """Load model with Colab-specific optimizations."""
        if self.model_loaded:
            return True
            
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded!")
            
            print("📥 Loading model...")
            
            if use_quantization and torch.cuda.is_available():
                print("🔧 Using 4-bit quantization for optimal memory usage")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16
                )
            else:
                print("🔧 Loading without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            print("✅ Model loaded successfully!")
            
            # Show memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"📊 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try reducing memory usage or using CPU")
            return False
    
    def generate_text(self, prompt, max_new_tokens=200, temperature=0.7, do_sample=True):
        """Generate text with Colab optimizations."""
        if not self.model_loaded:
            print("❌ Model not loaded!")
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    attention_mask=inputs.get('attention_mask')
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Clean up GPU memory
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return response
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
    
    def interactive_demo(self):
        """Interactive demo optimized for Colab."""
        print("\n🔄 Starting Interactive Demo")
        print("💡 Perfect for Google Colab!")
        print("💡 Try different European languages")
        print("💡 Type 'quit' to exit, 'clear' to reset")
        print("-" * 60)
        
        conversation_history = ""
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = ""
                    print("🧹 Conversation cleared!")
                    continue
                
                if user_input.lower() == 'memory':
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"📊 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
                    continue
                
                if not user_input:
                    continue
                
                # Build context-aware prompt
                if conversation_history:
                    prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                
                # Keep conversation manageable for Colab
                if len(prompt) > 1000:
                    lines = prompt.split('\n')
                    prompt = '\n'.join(lines[-6:])
                
                print("🤖 Generating response...")
                response = self.generate_text(prompt, max_new_tokens=150, temperature=0.8)
                
                if response:
                    print(f"\n🤖 TildeOpen: {response}")
                    conversation_history = f"{prompt} {response}"
                else:
                    print("❌ Failed to generate response")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_tests(self):
        """Run quick tests to verify the model works."""
        print("\n🧪 Running model tests...")
        
        test_prompts = [
            "Hello, I am TildeOpen and I can",
            "Bonjour, je suis un assistant IA qui peut",  # French
            "Hola, soy un asistente de IA que puede",     # Spanish
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            response = self.generate_text(prompt, max_new_tokens=30)
            if response:
                print(f"✅ Response: {response}")
            else:
                print("❌ Test failed")
        
        print("\n✅ Testing complete!")
    
    def cleanup(self):
        """Clean up memory."""
        print("🧹 Cleaning up memory...")
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Memory cleaned!")

def main():
    """Main function for Colab usage."""
    print("🌟 TildeOpen-30b - Google Colab Edition")
    print("=" * 60)
    
    # Initialize model
    model = ColabTildeModel()
    
    # Check environment
    is_colab = model.check_colab_environment()
    
    # Install dependencies if needed
    if is_colab:
        model.install_dependencies()
    
    try:
        # Load model
        print("\n🔄 Loading model (this may take a few minutes)...")
        if not model.load_model_colab():
            print("❌ Failed to load model")
            return
        
        # Run tests
        model.run_tests()
        
        # Start interactive demo
        print("\n🎉 Model ready! Starting interactive demo...")
        model.interactive_demo()
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
    
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()
