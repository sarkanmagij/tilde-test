#!/usr/bin/env python3
"""
Lightweight TildeOpen-30b Model Interface
Uses the model from Hugging Face without downloading the full 60GB+ files locally.
This approach streams model weights as needed and uses various optimization techniques.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class LightweightTildeModel:
    def __init__(self, model_name="TildeAI/TildeOpen-30b"):
        """
        Initialize the lightweight TildeOpen model.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing Lightweight TildeOpen-30b...")
        print(f"📱 Device: {self.device}")
        
    def load_model_lightweight(self, load_in_8bit=True, load_in_4bit=False):
        """
        Load the model with various optimization techniques to reduce memory usage.
        
        Args:
            load_in_8bit (bool): Use 8-bit quantization (requires less memory)
            load_in_4bit (bool): Use 4-bit quantization (even less memory, but may affect quality)
        """
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False,
                cache_dir=None  # Don't cache locally to save space
            )
            print("✅ Tokenizer loaded!")
            
            print("📥 Loading model with optimizations...")
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if load_in_4bit:
                print("🔧 Using 4-bit quantization (maximum memory savings)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                print("🔧 Using 8-bit quantization (balanced performance/memory)")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Model loading configuration
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
                "cache_dir": None,  # Don't cache locally
            }
            
            # Add quantization if specified
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            print("✅ Model loaded successfully with optimizations!")
            print(f"💾 Model is using: {self.device}")
            
            # Print memory usage if CUDA is available
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"📊 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try running with load_in_4bit=True for maximum memory savings")
            return False
    
    def generate_text(self, prompt, max_new_tokens=256, temperature=0.7, do_sample=True):
        """
        Generate text using the loaded model.
        
        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model_lightweight() first.")
        
        print(f"🤖 Generating response...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate with memory-efficient settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for efficiency
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
    
    def quick_test(self):
        """Quick test of the model with a simple prompt."""
        test_prompts = [
            "Hello, I am TildeOpen and I can help you with",
            "The weather today is",
            "Artificial intelligence is"
        ]
        
        print("\n🧪 Running quick tests...")
        for i, prompt in enumerate(test_prompts[:1]):  # Just test one to save time
            print(f"\nTest {i+1}: {prompt}")
            response = self.generate_text(prompt, max_new_tokens=50)
            if response:
                print(f"✅ Response: {response}")
            else:
                print("❌ Failed to generate response")
    
    def interactive_chat(self):
        """Lightweight interactive chat."""
        print("\n🔄 Starting lightweight chat with TildeOpen-30b")
        print("💡 Type 'quit', 'exit', or 'q' to end")
        print("💡 Type 'memory' to check GPU memory usage")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'memory':
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"📊 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
                    else:
                        print("📊 Using CPU - no GPU memory tracking")
                    continue
                
                if not user_input:
                    continue
                
                # Simple prompt format
                prompt = f"Human: {user_input}\nAssistant:"
                response = self.generate_text(prompt, max_new_tokens=200)
                
                if response:
                    print(f"\n🤖 TildeOpen: {response}")
                else:
                    print("❌ Failed to generate response")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function with optimization options."""
    print("🌟 Lightweight TildeOpen-30b Interface")
    print("=" * 50)
    
    # Check system capabilities
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        
        # Recommend quantization based on available memory
        if gpu_memory < 24:
            print("⚠️  Limited GPU memory detected - will use 4-bit quantization")
            use_4bit = True
            use_8bit = False
        elif gpu_memory < 48:
            print("💡 Using 8-bit quantization for optimal performance/memory balance")
            use_4bit = False
            use_8bit = True
        else:
            print("🚀 Sufficient GPU memory - using standard precision")
            use_4bit = False
            use_8bit = False
    else:
        print("⚠️  No GPU detected - this will be very slow with CPU")
        print("💡 Consider using Google Colab or a cloud GPU service")
        response = input("Continue with CPU? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
        use_4bit = True  # Use quantization to help with CPU
        use_8bit = False
    
    # Initialize model
    model = LightweightTildeModel()
    
    print("\n🔄 Loading model with optimizations...")
    if not model.load_model_lightweight(load_in_8bit=use_8bit, load_in_4bit=use_4bit):
        print("❌ Failed to load model")
        sys.exit(1)
    
    # Quick test
    model.quick_test()
    
    # Interactive chat
    try:
        model.interactive_chat()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
