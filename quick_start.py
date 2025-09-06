#!/usr/bin/env python3
"""
Quick Start Script for TildeOpen-30b
Perfect for M2 Mac users - provides immediate options without heavy downloads.
"""

import os
import sys
import platform

def print_banner():
    """Print welcome banner."""
    print("🌟" * 30)
    print("   TildeOpen-30b Quick Start")
    print("🌟" * 30)
    print()

def detect_system():
    """Detect system capabilities."""
    system_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version()
    }
    
    print(f"📱 System: {system_info['platform']} {system_info['machine']}")
    print(f"🐍 Python: {system_info['python_version']}")
    
    # Check for M1/M2 Mac
    is_apple_silicon = (
        system_info['platform'] == 'Darwin' and 
        system_info['machine'] == 'arm64'
    )
    
    if is_apple_silicon:
        print("🍎 Apple Silicon (M1/M2) detected!")
        return 'apple_silicon'
    elif system_info['platform'] == 'Darwin':
        print("🍎 Intel Mac detected")
        return 'intel_mac'
    elif system_info['platform'] == 'Linux':
        print("🐧 Linux detected")
        return 'linux'
    elif system_info['platform'] == 'Windows':
        print("🪟 Windows detected")
        return 'windows'
    else:
        return 'unknown'

def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"🎮 CUDA Available: {has_cuda}")
        if has_cuda:
            print(f"📱 GPU: {torch.cuda.get_device_name()}")
        return has_cuda
    except ImportError:
        print("⚠️ PyTorch not installed - can't check GPU")
        return False

def show_options(system_type, has_gpu):
    """Show recommended options based on system."""
    print("\n🚀 Recommended Options for Your System:")
    print("=" * 50)
    
    if system_type == 'apple_silicon':
        print("🎯 PERFECT FOR YOUR M2 MAC:")
        print()
        print("1. 📓 Google Colab (RECOMMENDED)")
        print("   ✅ Free GPU access")
        print("   ✅ No local storage needed")
        print("   ✅ Full model capabilities")
        print("   🔗 Open: https://colab.research.google.com/")
        print()
        
        print("2. 📡 API Demo (INSTANT)")
        print("   ✅ Zero setup required")
        print("   ✅ No downloads")
        print("   ✅ Works immediately")
        print("   ▶️  Run: python3 tilde_api_demo.py")
        print()
        
        print("3. 🚀 Minimal Demo (LOCAL)")
        print("   ✅ Small download (~2MB)")
        print("   ✅ Explore capabilities")
        print("   ✅ Check requirements")
        print("   ▶️  Run: python3 tilde_minimal.py")
        print()
        
        print("❌ NOT RECOMMENDED for M2 Mac:")
        print("   - Local model loading (requires 32GB+ RAM)")
        print("   - CPU inference (extremely slow)")
        
    elif has_gpu:
        print("🎮 GPU DETECTED - You have good options:")
        print()
        print("1. 🎯 Lightweight Local (RECOMMENDED)")
        print("   ▶️  Run: python3 tilde_lightweight.py")
        print()
        print("2. 📓 Google Colab (ALTERNATIVE)")
        print("   🔗 Open: https://colab.research.google.com/")
        print()
        print("3. 📡 API Demo (QUICK TEST)")
        print("   ▶️  Run: python3 tilde_api_demo.py")
        
    else:
        print("💻 CPU ONLY SYSTEM:")
        print()
        print("1. 📓 Google Colab (HIGHLY RECOMMENDED)")
        print("   ✅ Free GPU access")
        print("   🔗 Open: https://colab.research.google.com/")
        print()
        print("2. 📡 API Demo (INSTANT)")
        print("   ✅ Zero setup")
        print("   ▶️  Run: python3 tilde_api_demo.py")
        print()
        print("3. 🚀 Minimal Demo (EXPLORE)")
        print("   ✅ Small download")
        print("   ▶️  Run: python3 tilde_minimal.py")
        print()
        print("⚠️  Local model loading not recommended (very slow)")

def show_cloud_options():
    """Show cloud deployment options."""
    print("\n☁️ Cloud Deployment Options:")
    print("=" * 30)
    print()
    print("📓 Google Colab:")
    print("   - Free GPU access")
    print("   - Jupyter notebook interface")
    print("   - Perfect for experimentation")
    print()
    print("🤗 Hugging Face Spaces:")
    print("   - Deploy web interface")
    print("   - Share with others")
    print("   - Production ready")
    print()
    print("📱 Kaggle Notebooks:")
    print("   - Free GPU hours")
    print("   - Competition platform")
    print("   - Community features")

def launch_option():
    """Let user choose and launch an option."""
    print("\n🎯 Quick Launch:")
    print("=" * 20)
    print("1. 📡 API Demo (instant)")
    print("2. 🚀 Minimal Demo (small download)")
    print("3. 📓 Open Google Colab")
    print("4. ℹ️  More information")
    print("5. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🔄 Launching API Demo...")
                os.system("python3 tilde_api_demo.py")
                break
            elif choice == '2':
                print("\n🔄 Launching Minimal Demo...")
                os.system("python3 tilde_minimal.py")
                break
            elif choice == '3':
                print("\n🔄 Opening Google Colab...")
                import webbrowser
                webbrowser.open("https://colab.research.google.com/github/sarkanmagij/tilde-test/blob/main/TildeOpen_30b_Colab.ipynb")
                print("📓 Colab should open in your browser!")
                break
            elif choice == '4':
                print("\n📚 For more information:")
                print("   - README.md: Complete documentation")
                print("   - DEPLOYMENT.md: Cloud deployment guide")
                print("   - GitHub: https://github.com/sarkanmagij/tilde-test")
                continue
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    print_banner()
    
    # Detect system
    system_type = detect_system()
    has_gpu = check_gpu()
    
    # Show recommendations
    show_options(system_type, has_gpu)
    
    # Show cloud options
    show_cloud_options()
    
    # Launch option
    launch_option()

if __name__ == "__main__":
    main()
