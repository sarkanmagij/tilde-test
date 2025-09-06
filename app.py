#!/usr/bin/env python3
"""
TildeOpen-30b Gradio Interface for Hugging Face Spaces
A streamlined web interface for interacting with the TildeOpen-30b model.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class TildeSpaceModel:
    def __init__(self):
        self.model_name = "TildeAI/TildeOpen-30b"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
    def load_model(self):
        """Load the model with optimizations for HF Spaces."""
        if self.model_loaded:
            return True
            
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                trust_remote_code=True
            )
            
            print("📥 Loading model...")
            
            # Use quantization if on GPU
            if torch.cuda.is_available():
                print("🔧 Using 4-bit quantization for GPU")
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
                    low_cpu_mem_usage=True
                )
            else:
                print("🔧 Loading on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
        """Generate text response."""
        if not self.model_loaded:
            if not self.load_model():
                return "❌ Failed to load model. Please try again."
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                del outputs
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Initialize the model
model_interface = TildeSpaceModel()

def chat_interface(message, history, max_tokens, temperature, top_p):
    """Gradio chat interface function."""
    if not message.strip():
        return history, ""
    
    # Build conversation context
    if history:
        # Format conversation history
        context = ""
        for human, assistant in history:
            context += f"Human: {human}\nAssistant: {assistant}\n"
        prompt = f"{context}Human: {message}\nAssistant:"
    else:
        prompt = f"Human: {message}\nAssistant:"
    
    # Keep context manageable
    if len(prompt) > 1500:
        lines = prompt.split('\n')
        prompt = '\n'.join(lines[-8:])
    
    # Generate response
    response = model_interface.generate_response(
        prompt, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p
    )
    
    # Update history
    history.append((message, response))
    
    return history, ""

def create_demo():
    """Create the Gradio interface."""
    
    # Custom CSS for better appearance
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .chat-message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
    }
    """
    
    with gr.Blocks(
        title="TildeOpen-30b Chat",
        theme=gr.themes.Soft(),
        css=css
    ) as demo:
        
        gr.Markdown("""
        # 🌍 TildeOpen-30b Chat Interface
        
        Chat with **TildeOpen-30b**, a 30B parameter model optimized for **34 European languages**!
        
        **Features:**
        - 🔓 Open Source (CC-BY-4.0)
        - 🌍 34 European languages supported
        - ⚡ Optimized for Nordic and Eastern European languages
        - 🏗️ Enterprise-ready model trained on LUMI supercomputer
        
        **Try different languages:** English, French, German, Spanish, Polish, Russian, Italian, and more!
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    show_label=False,
                    avatar_images=("👤", "🤖")
                )
                
                msg = gr.Textbox(
                    placeholder="Type your message here... (Try different European languages!)",
                    show_label=False,
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Generation Settings")
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=512,
                    value=256,
                    step=10,
                    label="Max Tokens",
                    info="Maximum length of response"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p",
                    info="Nucleus sampling parameter"
                )
                
                gr.Markdown("""
                ### 💡 Examples
                - Hello, how are you?
                - Comment allez-vous? (French)
                - ¿Cómo estás? (Spanish)
                - Как дела? (Russian)
                - Wie geht es dir? (German)
                - Come stai? (Italian)
                - Jak się masz? (Polish)
                """)
        
        # Event handlers
        def submit_message(message, history, max_tokens, temperature, top_p):
            return chat_interface(message, history, max_tokens, temperature, top_p)
        
        def clear_conversation():
            return [], ""
        
        # Set up interactions
        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            submit_message,
            inputs=[msg, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, msg]
        )
        
        gr.Markdown("""
        ---
        ### 📚 About TildeOpen-30b
        
        **Supported Languages (34 total):** Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hungarian, Icelandic, Irish, Italian, Latvian, Lithuanian, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovene, Spanish, Swedish, Turkish, Ukrainian, and more.
        
        **Links:**
        - [Model on Hugging Face](https://huggingface.co/TildeAI/TildeOpen-30b)
        - [Tilde.ai](https://tilde.ai/tildeopen-llm/)
        - [GitHub Repository](https://github.com/sarkanmagij/tilde-test)
        
        *Note: This is a foundational model not yet adapted for instruction following or safety alignment.*
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
