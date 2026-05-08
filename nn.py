from llama_cpp import Llama
import random
from huggingface_hub import hf_hub_download
import json
import os


class NN:
    def __init__(self, repo_id, filename, use_gpu=False, n_ctx=32768, 
                 max_history_len=10, reset_history_count=4, 
                 save_history_count=0, history_file_path=None):
        """
        Initialize the neural network
        :param repo_id: Hugging Face repository ID
        :param filename: Model filename in the repository
        :param use_gpu: Whether to use GPU acceleration
        :param n_ctx: Context size
        :param max_history_len: Maximum history length
        :param reset_history_count: History reset count threshold
        :param save_history_count: History save count threshold
        :param history_file_path: Path to history file
        """
        self.history = []
        self.reset_history_count = 0
        self.save_history_count = 0
        self.max_history_len = 0

        #hf_token=""
        #with open(token_path) as tokenf: hf_token=tokenf.read().strip() 
        self.history_file_path = history_file_path

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
        """
        Save the chat history to file
        """
        with open(self.history_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f)

    def load_history(self, history_file_path=None):
        """
        Load chat history from file
        :param history_file_path: Path to history file (optional)
        """
        if not history_file_path and self.history_file_path: 
            history_file_path = self.history_file_path
        elif not history_file_path: 
            return
            
        if os.path.exists(history_file_path):
            with open(history_file_path, encoding='utf-8') as f:
                self.history = json.load(f)
                self.llm.reset()

    def clip_history(self):
        """
        Clip the chat history to prevent exceeding limits
        """
        if len(self.history) >= self.max_history_len:
            while len(self.history) > self.save_history_count and len(self.history) > self.max_history_len - self.reset_history_count:
                self.history.pop(2)
                self.history.pop(2)
            self.llm.reset()
    
    def chat(self, objects, max_new_tokens=128, role="user", save_chat=True):
        """
        Perform a chat interaction
        :param objects: Input text or objects
        :param max_new_tokens: Maximum number of tokens to generate
        :param role: Role for the message
        :param save_chat: Whether to save the chat to history
        :return: Generated response
        """
        self.clip_history()

        self.history.append({
            "role": role,
            "content": objects
        })
        
        response = self.llm.create_chat_completion(
            self.history, 
            temperature=0.5, 
            max_tokens=max_new_tokens
        )
        
        self.history.append(response['choices'][0]['message'])
        
        if save_chat and self.history_file_path: 
            self.save_history()
            
        return response['choices'][0]['message']['content']
    
    def chat_async(self, objects, max_new_tokens=128, role="user", save_chat=True):
        """
        Perform an asynchronous chat interaction
        :param objects: Input text or objects
        :param max_new_tokens: Maximum number of tokens to generate
        :param role: Role for the message
        :param save_chat: Whether to save the chat to history
        :return: Generator yielding response tokens
        """
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
        if save_chat and self.history_file_path: 
            self.save_history()

    def chat_no_history(self, objects, max_new_tokens=128, role="user"):
        """
        Perform a chat interaction without keeping history
        :param objects: Input text or objects
        :param max_new_tokens: Maximum number of tokens to generate
        :param role: Role for the message
        :return: Generated response
        """
        self.llm.reset()
        response = self.llm.create_chat_completion(
            [{"role": role, "content": objects}], 
            temperature=0.5, 
            max_tokens=max_new_tokens
        )
        
        return response['choices'][0]['message']['content']