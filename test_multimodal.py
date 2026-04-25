"""
Test script for multimodal functionality with Gemma 4
This script demonstrates how to use image processing with the NN class
"""
from config_manager import ConfigManager
cm=ConfigManager()
cm.load_config()
cm.set_hf_env()
config=cm.get_json_config()

from nn import NN
from screen_capture import ScreenCapture
import base64

def test_multimodal():
    print("Testing multimodal functionality...")
    
    # Initialize the model with multimodal support
    print("\n1. Initializing model with mmproj...")
    neuro = NN(
        repo_id=config["model"]["repo_id"],
        filename=config["model"]["filename"],
        mmproj_filename=config["model"]["mmproj_filename"],
        use_gpu=True,
        n_ctx=8192
    )
    
    # Test 1: Text-only chat (should still work)
    print("\n2. Testing text-only chat...")
    response = neuro.chat_no_history("Hello! Can you introduce yourself?")
    print(f"Response: {response[:100]}...")
    
    # Test 2: Image + text chat
    print("\n3. Testing image + text chat...")
    try:
        # Capture screen
        screen_capture = ScreenCapture()
        base64_image = screen_capture.capture_to_base64(scale_factor=2)
        screen_capture.save_base64_image(base64_image)
        
        # Create message with image and text
        content = [
            neuro.make_text_object("опиши, что ты видишь? ответь 5 словами."),
            neuro.make_image_object(base64_image)
        ]
        
        print("Sending image to model...")
        response = neuro.chat(content, max_new_tokens=256)
        print(f"\nModel's description:\n{response}")
        
    except Exception as e:
        print(f"Error during image test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Test completed!")

if __name__ == "__main__":
    #test_multimodal()
    neuro = NN(
        repo_id=config["model"]["repo_id"],
        filename=config["model"]["filename"],
        mmproj_filename=config["model"]["mmproj_filename"],
        use_gpu=True,
        n_ctx=8192
    )
    while True:
        inp=input(">>")
        screen_capture = ScreenCapture()
        base64_image = screen_capture.capture_to_base64(scale_factor=2)
        screen_capture.save_base64_image(base64_image)
        
        # Create message with image and text
        content = [
            neuro.make_text_object(inp),
            neuro.make_image_object(base64_image)
        ]
        
        print("Sending image to model...")
        response = neuro.chat(content, max_new_tokens=256)
        print(f"\nModel's description:\n{response}")
