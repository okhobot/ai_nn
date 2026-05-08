from llama_cpp import Llama
import random
from huggingface_hub import hf_hub_download
import json
import os

class NN:
    llm = None
    history = []
    reset_history_count = 0
    save_history_count = 0
    max_history_len = 0
    
    def __init__(self, repo_id, filename, use_gpu=False, n_ctx=32768, max_history_len=10, reset_history_count=4, save_history_count=0, history_file_path=None): 
        #hf_token=""
        #with open(token_path) as tokenf: hf_token=tokenf.read().strip() 
        self.history_file_path=history_file_path

        self.max_history_len = max(2, max_history_len)
        self.reset_history_count = max(save_history_count, reset_history_count)
        self.save_history_count = save_history_count

        model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename
        )

        self.llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1 if use_gpu else 0,
        n_threads=8,
        n_ctx=n_ctx,
        verbose=False
        )

    def save_history(self):
        with open(self.history_file_path, 'w') as f:
            #print("S",self.history)
            json.dump(self.history, f)
    def load_history(self, history_file_path=None):
        if not history_file_path and self.history_file_path: history_file_path=self.history_file_path
        elif not history_file_path: return
        #print(history_file_path)
        if os.path.exists(history_file_path):
            with open(history_file_path) as f:
                self.history = json.load(f)
                self.llm.reset()
                #print("L",self.history)


    def clip_history(self):
        if len(self.history) >= self.max_history_len:
            while len(self.history) > self.save_history_count and len(self.history) > self.max_history_len - self.reset_history_count:
                self.history.pop(2)
                self.history.pop(2)
            self.llm.reset()
    def chat(self, objects, max_new_tokens=128, role="user", save_chat=True):
        self.clip_history()
        #print(self.history)

        self.history.append({
                "role": role,
                "content": objects
            })
        response = self.llm.create_chat_completion(self.history, temperature=0.5, max_tokens=max_new_tokens)
        self.history.append(response['choices'][0]['message'])
        if save_chat and self.history_file_path: self.save_history()
        return response['choices'][0]['message']['content']
    
    def chat_async(self, objects, max_new_tokens=128, role="user", save_chat=True):
        self.clip_history()
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
        self.history.append(accumulated_message)
        if save_chat and self.history_file_path: self.save_history()
        


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