from llama_cpp import Llama
from llama_cpp.llama_chat_format import NanoLlavaChatHandler, MoondreamChatHandler
import random
import os
from huggingface_hub import hf_hub_download

class NN:
    llm = None
    chat_handler = None
    history = []
    reset_history_count = 0
    save_history_count = 0
    max_history_len = 0
    
    def __init__(self, repo_id, filename, use_gpu=False, n_ctx=32768, max_history_len=10, reset_history_count=4, save_history_count=0, mmproj_filename=None): 
        #hf_token=""
        #with open(token_path) as tokenf: hf_token=tokenf.read().strip() 
        
        self.max_history_len = max(4, max_history_len)
        self.reset_history_count = max(save_history_count, reset_history_count)
        self.save_history_count = save_history_count

        print(f"[NN] Downloading model from {repo_id}/{filename}")
        print(f"[NN] HF_HOME environment: {os.environ.get('HF_HOME', 'NOT SET')}")
        print(f"[NN] HUGGINGFACE_HUB_CACHE environment: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'NOT SET')}")
        
        model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        )
        
        print(f"[NN] Model downloaded to: {model_path}")

        # Initialize chat handler for multimodal support if mmproj file is provided
        if mmproj_filename:
            try:
                print(f"[NN] Downloading mmproj from {repo_id}/{mmproj_filename}")
                mmproj_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=mmproj_filename
                )
                print(f"[NN] Multimodal projector downloaded to: {mmproj_path}")
                print(f"Loading multimodal projector from: {mmproj_path}")
                self.chat_handler = NanoLlavaChatHandler(
                    clip_model_path=mmproj_path,
                    
                    #n_gpu_layers=-1 if use_gpu else 0,
                    verbose=False
                )
                print("Multimodal projector loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load multimodal projector: {e}")
                print("Model will run in text-only mode")
                self.chat_handler = None
        else:
            print("No mmproj file specified. Running in text-only mode.")
            self.chat_handler = None

        self.llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1 if use_gpu else 0,
        n_threads=8,
        n_ctx=n_ctx,
        chat_handler=self.chat_handler,
        verbose=False
        )


    def clip_history(self):
        if len(self.history) >= self.max_history_len:
            while len(self.history) > self.save_history_count and len(self.history) > self.max_history_len - self.reset_history_count:
                self.history.pop(2)
                self.history.pop(2)
            self.llm.reset()

    def replace_objects_in_history(self, objects):
        text_only = None
        if isinstance(objects, list):
            text_only = [obj for obj in objects if isinstance(obj, dict) and obj.get('type') == 'text']
        self.history[-1] = {
            "role": self.history[-1]["role"],
            "content": text_only if text_only else objects
        }
    def chat(self, objects, max_new_tokens=128, role="user"):
        self.clip_history()

        self.history.append({
            "role": role,
            "content": objects
        })
        
        response = self.llm.create_chat_completion(self.history, temperature=0.5, max_tokens=max_new_tokens)
        
        self.replace_objects_in_history(objects)
        
        self.history.append(response['choices'][0]['message'])
        #print(self.history)
        return response['choices'][0]['message']['content']
    
    def chat_async(self, objects, max_new_tokens=128, role="user"):
        self.clip_history()
        

        self.history.append({
            "role": role,
            "content": objects
        })
        
        # Create a streaming completion request
        response = self.llm.create_chat_completion(
            self.history, 
            temperature=0.5, 
            max_tokens=max_new_tokens,
            stream=True
        )
        
        # Initialize the message to accumulate the streamed content
        accumulated_message = {"role": "assistant", "content": ""}
        full_response = ""
        
        # Yield each token as it arrives
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content = delta['content']
                full_response += content
                yield content  # Stream the content as it's generated
        
        # After streaming is done, append the complete message to history
        accumulated_message['content'] = full_response
        
        #self.replace_objects_in_history(objects)
        
        self.history.append(accumulated_message)

    def chat_no_history(self, objects, max_new_tokens=128, role="user"):
        self.llm.reset()
        response = self.llm.create_chat_completion([{"role": role, "content": objects}], temperature=0.5, max_tokens=max_new_tokens)        
        self.llm.reset()

        return response['choices'][0]['message']['content']

    def make_text_object(self, text):
        return {
            "type": "text",
            "text": text
        }
    def make_image_object(self, img):
        return {
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{img}"}
            }