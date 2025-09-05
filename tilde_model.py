#!/usr/bin/env python3
"""
TildeOpen-30b Model Connection Script
Connects to and interacts with the TildeAI/TildeOpen-30b model via Hugging Face transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import warnings

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class TildeOpenModel:
    def __init__(self, model_name="TildeAI/TildeOpen-30b"):
        """
        Initialize the TildeOpen model.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing TildeOpen-30b model...")
        print(f"📱 Device: {self.device}")
        
    def load_model(self):
        """Load the tokenizer and model."""
        try:
            print("📥 Loading tokenizer...")
            # Important: use_fast=False is required for this model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False
            )
            
            print("📥 Loading model (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_text(self, prompt, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        Generate text using the loaded model.
        
        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (0.0 = deterministic)
            do_sample (bool): Whether to use sampling or greedy decoding
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"🤖 Generating response for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the output
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return None
    
    def chat_loop(self):
        """Interactive chat loop with the model."""
        print("\n🔄 Starting interactive chat with TildeOpen-30b")
        print("💡 Type 'quit', 'exit', or 'q' to end the session")
        print("💡 Type 'clear' to clear the conversation")
        print("-" * 50)
        
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
                
                if not user_input:
                    continue
                
                # Build prompt with conversation history
                if conversation_history:
                    prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                
                # Generate response
                response = self.generate_text(prompt, max_new_tokens=256, temperature=0.8)
                
                if response:
                    print(f"\n🤖 TildeOpen: {response}")
                    # Update conversation history
                    conversation_history = f"{prompt} {response}"
                else:
                    print("❌ Failed to generate response")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in chat loop: {e}")

def main():
    """Main function to demonstrate model usage."""
    print("🌟 TildeOpen-30b Model Connection Script")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎮 CUDA available: {torch.cuda.get_device_name()}")
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("⚠️  CUDA not available, using CPU (this will be slow for a 30B model)")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Initialize and load model
    model_interface = TildeOpenModel()
    
    if not model_interface.load_model():
        print("❌ Failed to load model. Exiting...")
        sys.exit(1)
    
    # Test with a simple prompt
    print("\n🧪 Testing model with a simple prompt...")
    test_prompt = "Hello, how are you today?"
    response = model_interface.generate_text(test_prompt, max_new_tokens=100)
    
    if response:
        print(f"✅ Test successful!")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
    else:
        print("❌ Test failed")
        sys.exit(1)
    
    # Start interactive chat
    try:
        model_interface.chat_loop()
    except Exception as e:
        print(f"❌ Error in main: {e}")

if __name__ == "__main__":
    main()
