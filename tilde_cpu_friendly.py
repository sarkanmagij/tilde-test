#!/usr/bin/env python3
"""
CPU-Friendly TildeOpen-30b Model Interface
Uses the model from Hugging Face with CPU optimizations and without full local download.
This approach is designed to work without CUDA and minimizes memory usage.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import sys
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class CPUFriendlyTildeModel:
    def __init__(self, model_name="TildeAI/TildeOpen-30b"):
        """
        Initialize the CPU-friendly TildeOpen model.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        print(f"🚀 Initializing CPU-Friendly TildeOpen-30b...")
        print(f"💻 Running on CPU with memory optimizations")
        
    def load_model_cpu_optimized(self):
        """
        Load the model with CPU-specific optimizations.
        """
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False,
                cache_dir=None  # Don't cache locally
            )
            print("✅ Tokenizer loaded!")
            
            print("📥 Loading model with CPU optimizations...")
            print("⚠️  Note: This is a 30B parameter model - it will use significant RAM and be slow on CPU")
            print("🔄 Loading... (this may take several minutes)")
            
            # CPU-optimized loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for better CPU compatibility
                device_map="cpu",  # Force CPU usage
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Important for large models
                cache_dir=None,  # Don't cache locally to save disk space
                # Note: We're not using torch_compile as it can be problematic on some systems
            )
            
            print("✅ Model loaded successfully!")
            
            # Force garbage collection to free up memory
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 This model is very large (30B parameters) and may exceed available RAM")
            return False
    
    def generate_text(self, prompt, max_new_tokens=100, temperature=0.8, do_sample=True):
        """
        Generate text using the loaded model with CPU optimizations.
        
        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum number of new tokens to generate (kept small for CPU)
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model_cpu_optimized() first.")
        
        print(f"🤖 Generating response (this may take a while on CPU)...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate with CPU-friendly settings
            with torch.no_grad():  # Disable gradients to save memory
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for efficiency
                    num_beams=1,  # Use greedy search for speed
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Clean up
            del outputs
            gc.collect()
            
            return response
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
    
    def quick_test(self):
        """Quick test with a very simple prompt."""
        print("\n🧪 Running a quick test...")
        test_prompt = "Hello"
        print(f"Test prompt: '{test_prompt}'")
        
        response = self.generate_text(test_prompt, max_new_tokens=20)
        if response:
            print(f"✅ Response: {response}")
            return True
        else:
            print("❌ Test failed")
            return False
    
    def simple_chat(self):
        """Simple chat interface optimized for CPU usage."""
        print("\n🔄 Starting simple chat with TildeOpen-30b")
        print("💡 Type 'quit', 'exit', or 'q' to end")
        print("⚠️  Responses will be slow on CPU - please be patient")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Simple prompt - keep it short for CPU efficiency
                response = self.generate_text(user_input, max_new_tokens=50)
                
                if response:
                    print(f"\n🤖 TildeOpen: {response}")
                else:
                    print("❌ Failed to generate response")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def check_system_requirements():
    """Check if the system can handle the model."""
    import psutil
    
    # Get available RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💾 Available RAM: {ram_gb:.1f}GB")
    
    if ram_gb < 32:
        print("⚠️  WARNING: Less than 32GB RAM detected")
        print("   The 30B parameter model may not fit in memory")
        print("   Consider using a smaller model or cloud GPU service")
        return False
    elif ram_gb < 64:
        print("⚠️  CAUTION: Limited RAM detected")
        print("   The model may load but performance will be very slow")
    else:
        print("✅ Sufficient RAM for model loading")
    
    return True

def main():
    """Main function with system checks."""
    print("🌟 CPU-Friendly TildeOpen-30b Interface")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("💡 Consider using Google Colab with GPU or a cloud service")
            print("💡 Or try a smaller model like TinyLlama or Phi-2")
            sys.exit(1)
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("• This is a 30B parameter model - it's VERY large")
    print("• CPU inference will be extremely slow (minutes per response)")
    print("• Consider using Google Colab with free GPU instead")
    print("• The model streams from HuggingFace (no 60GB local download)")
    
    response = input("\nProceed with CPU loading? (y/N): ")
    if response.lower() != 'y':
        print("💡 Recommended: Use 'python3 tilde_lightweight.py' on a GPU system")
        sys.exit(0)
    
    # Initialize model
    model = CPUFriendlyTildeModel()
    
    print(f"\n🔄 Loading model (this will take several minutes)...")
    if not model.load_model_cpu_optimized():
        print("❌ Failed to load model")
        print("💡 Try using a cloud GPU service or a smaller model")
        sys.exit(1)
    
    # Quick test
    if model.quick_test():
        print("\n🎉 Model loaded and tested successfully!")
        
        # Simple chat
        try:
            model.simple_chat()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Model test failed")

if __name__ == "__main__":
    main()
