#!/usr/bin/env python3
"""
Quick test script to verify TildeOpen-30b model connection.
This script loads the model and performs a simple generation test.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def test_model_connection():
    """Test connection to TildeOpen-30b model."""
    print("🌟 Testing TildeOpen-30b Model Connection")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("⚠️  No GPU detected - using CPU (will be slow for 30B model)")
    
    try:
        print("\n📥 Loading tokenizer...")
        # Load tokenizer (use_fast=False is required for this model)
        tokenizer = AutoTokenizer.from_pretrained("TildeAI/TildeOpen-30b", use_fast=False)
        print("✅ Tokenizer loaded successfully!")
        
        print("\n📥 Loading model (this may take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "TildeAI/TildeOpen-30b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully!")
        
        # Test generation
        print("\n🧪 Testing text generation...")
        test_prompt = "Hello, I am TildeOpen and I can help you with"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(test_prompt):].strip()
        
        print(f"✅ Generation test successful!")
        print(f"📝 Prompt: {test_prompt}")
        print(f"🤖 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model_connection()
    if success:
        print("\n🎉 Model connection test passed!")
        print("💡 You can now use the full script: python3 tilde_model.py")
    else:
        print("\n❌ Model connection test failed!")
        print("💡 Check your internet connection and try again.")
