from llama_cpp import Llama
import random
from huggingface_hub import hf_hub_download

class NN:
    llm=None
    history=[]
    reset_history_count=0
    save_history_count=0
    max_history_len=0
    def __init__(self, repo_id, filename, cache_dir=None, hf_token=None, use_gpu=False, use_cache=True, n_ctx=32768, max_history_len=10, reset_history_count=4, save_history_count=0, offline=False): 
        #hf_token=""
        #with open(token_path) as tokenf: hf_token=tokenf.read().strip() 
        
        self.max_history_len=max(4,max_history_len)
        self.reset_history_count=max(save_history_count,reset_history_count)
        self.save_history_count=save_history_count

        model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=hf_token,
        local_files_only=offline 
        )

        self.llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1 if use_gpu else 0,
        n_threads=8,
        n_ctx=n_ctx,
        verbose=False,
        use_cache=use_cache
        )


    def chat(self, objects, max_new_tokens=128, role="user"):
        reset=False
        if len(self.history)>=self.max_history_len:
            while len(self.history)>self.save_history_count and len(self.history)>self.max_history_len-self.reset_history_count:
                self.history.pop(2)
                self.history.pop(2)
            self.llm.reset()

        self.history.append({
                "role": role,
                "content": objects
            })
        #print((self.history))
        response = self.llm.create_chat_completion(self.history,temperature=0.5, max_tokens=max_new_tokens)
        self.history.append(response['choices'][0]['message'])
        
        
        return response['choices'][0]['message']['content']
    
    def chat_no_history(self, objects, max_new_tokens=128, role="user"):
        self.llm.reset()
        response = self.llm.create_chat_completion({"role": role,"content": objects},temperature=0.5, max_tokens=max_new_tokens)        
        self.llm.reset()

        return response['choices'][0]['message']['content']
    

    def make_text_object(self, text):
        return {
            "type": "text",
			"text": text
        }
