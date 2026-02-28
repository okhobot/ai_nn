from llama_cpp import Llama
import random
class NN:
    llm=None
    history=[]
    reset_history_count=0
    max_history_len=0
    def __init__(self, repo_id, filename, use_gpu=False, n_ctx=32768, max_history_len=10, reset_history_count=4): 
        hf_token=""
        with open("token.txt") as tokenf:
            hf_token=tokenf.read().strip() 
        
        self.max_history_len=max(4,max_history_len)
        self.reset_history_count=max(2,reset_history_count)

        self.llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_gpu_layers=-1 if use_gpu else 0,  # явно просим CPU — будет CPU
        n_threads=8,
        n_ctx=n_ctx,#32768,        # уменьшите контекст для экономии памяти
        verbose=False,
        token=hf_token
        )

    def send_message(self, objects, role="user"):
        reset=False
        if len(self.history)>=self.max_history_len:
            while len(self.history)>2 and len(self.history)>self.max_history_len-self.reset_history_count:
                self.history.pop(2)
                self.history.pop(2)
            self.llm.reset()

        self.history.append({
                "role": role,
                "content": objects
            })
        #print((self.history))
        response = self.llm.create_chat_completion(self.history,temperature=0.5, max_tokens=128)
        self.history.append(response['choices'][0]['message'])
        
        
        return response['choices'][0]['message']['content']
    

    def make_text_object(self, text):
        return {
            "type": "text",
			"text": text
        }

