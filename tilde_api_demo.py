#!/usr/bin/env python3
"""
TildeOpen-30b Inference API Demo
Uses Hugging Face's Inference API - no local model download required!
This is the most lightweight approach - zero local storage and minimal requirements.
"""

import requests
import json
import time

class TildeInferenceAPI:
    def __init__(self, api_token=None):
        """
        Initialize the Inference API client.
        
        Args:
            api_token (str): Hugging Face API token (optional for public models)
        """
        self.model_name = "TildeAI/TildeOpen-30b"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {}
        
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
        
        print(f"🚀 TildeOpen-30b Inference API Client")
        print(f"🔗 Model: {self.model_name}")
        print(f"📡 API URL: {self.api_url}")
    
    def query_model(self, prompt, max_length=200, temperature=0.7, do_sample=True):
        """
        Query the model via Inference API.
        
        Args:
            prompt (str): Input text prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text or None if error
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "return_full_text": False  # Only return the generated part
            }
        }
        
        try:
            print(f"📡 Sending request to Hugging Face API...")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    return generated_text.strip()
                else:
                    print(f"❌ Unexpected response format: {result}")
                    return None
            elif response.status_code == 503:
                print("⏳ Model is loading on the server, please wait...")
                return "Model is currently loading. Please try again in a moment."
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def test_connection(self):
        """Test the API connection with a simple prompt."""
        print("\n🧪 Testing API connection...")
        test_prompt = "Hello, I am TildeOpen and I"
        
        result = self.query_model(test_prompt, max_length=100)
        
        if result:
            print(f"✅ API Test successful!")
            print(f"📝 Prompt: {test_prompt}")
            print(f"🤖 Response: {result}")
            return True
        else:
            print("❌ API test failed")
            return False
    
    def interactive_chat(self):
        """Interactive chat using the Inference API."""
        print("\n🔄 Starting API-based chat with TildeOpen-30b")
        print("💡 Type 'quit', 'exit', or 'q' to end")
        print("💡 Type 'help' for tips")
        print("🌐 Using Hugging Face Inference API - no local download!")
        print("-" * 60)
        
        conversation_history = ""
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n💡 Tips:")
                    print("• Keep prompts clear and specific")
                    print("• The model supports 34 languages")
                    print("• Try asking questions in different European languages")
                    print("• Response time depends on API load")
                    continue
                
                if user_input.lower() == 'clear':
                    conversation_history = ""
                    print("🧹 Conversation cleared!")
                    continue
                
                if not user_input:
                    continue
                
                # Build prompt with some context
                if conversation_history:
                    prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
                else:
                    prompt = f"Human: {user_input}\nAssistant:"
                
                # Keep conversation history manageable
                if len(prompt) > 1000:
                    # Keep only the last part of the conversation
                    lines = prompt.split('\n')
                    prompt = '\n'.join(lines[-6:])
                
                print("🔄 Generating response via API...")
                response = self.query_model(prompt, max_length=300, temperature=0.8)
                
                if response:
                    # Clean up the response
                    if response.startswith("Human:") or response.startswith("Assistant:"):
                        # Extract just the assistant's response
                        parts = response.split("Assistant:")
                        if len(parts) > 1:
                            response = parts[-1].strip()
                    
                    print(f"\n🤖 TildeOpen: {response}")
                    
                    # Update conversation history
                    conversation_history = f"{prompt} {response}"
                else:
                    print("❌ Failed to get response from API")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function for API demo."""
    print("🌟 TildeOpen-30b Inference API Demo")
    print("=" * 50)
    print("✨ Zero download required - uses Hugging Face API!")
    print("🚀 Lightweight - works on any system with internet")
    print("🌍 Supports 34 European languages")
    
    # Check for API token
    print(f"\n🔑 API Authentication:")
    print("💡 You can use this without an API token (rate limited)")
    print("💡 For better performance, get a free token at https://huggingface.co/settings/tokens")
    
    api_token = input("\n🔑 Enter your HuggingFace API token (or press Enter to skip): ").strip()
    if not api_token:
        print("📝 Using without authentication (may have rate limits)")
        api_token = None
    
    # Initialize API client
    api_client = TildeInferenceAPI(api_token)
    
    # Test connection
    if api_client.test_connection():
        print("\n🎉 Ready to chat!")
        
        # Start interactive chat
        try:
            api_client.interactive_chat()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("\n❌ Failed to connect to API")
        print("💡 Check your internet connection and try again")
        print("💡 The model might be loading - try again in a few minutes")

if __name__ == "__main__":
    main()
