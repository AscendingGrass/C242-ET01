from dataclasses import dataclass, asdict
from enum import Enum
from src.models.basellm import BaseLLM, ChatSession
import ollama
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

@dataclass
class OllamaConfig:
    model_name: str = "gemma:2b-instruct"
    system_instruction: str = None
    options: dict = None

"""
Available Options: 
{
    "num_keep": 5,
    "seed": 42,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "min_p": 0.0,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": true,
    "stop": ["\n", "user:"],
    "numa": false,
    "num_ctx": 1024,
    "num_batch": 2,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": false,
    "vocab_only": false,
    "use_mmap": true,
    "use_mlock": false,
    "num_thread": 8
}
"""
    



class OllamaModel(BaseLLM):
    def __init__(self, model_config:OllamaConfig=OllamaConfig()):
        super().__init__(model_name=model_config.model_name)
        self.options = model_config.options
        self.system_instruction = model_config.system_instruction
        

    def generate(self, prompt, return_response:bool=False):

        input_chatml = [dict(
            role="user",
            content=prompt
        )]

        if self.system_instruction is not None:
            input_chatml.insert(0, dict(
                role="system",
                content=self.system_instruction
            ))

        response = ollama.chat(model=self.model_name, messages=input_chatml, options=self.options)
        if return_response:
            return response
        
        return response['message']['content']
    
    def get_chat_session(self, history=[]):
        if self.system_instruction is not None:
            history.insert(0, {"role": "system", "content": self.system_instruction})
        return OllamaChatSession(self, history)
    

class OllamaChatSession(ChatSession):
    def __init__(self, model:OllamaModel, history=[]):
        self.model = model
        self.history = history

    def chat(self, message:str, return_response:bool=False):
        self.history.append(dict(
            role="user",
            content=message
        ))

        response = ollama.chat(model=self.model.model_name, messages=self.history, options=self.model.options)
        response_text = response['message']['content']

        self.history.append(dict(
            role="assistant",
            content=response_text
        ))

        if return_response:
            return response
        
        return response_text

    def get_history(self):
        return self.history