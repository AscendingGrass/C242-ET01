import sys

if sys.platform.lower() != "linux":
    raise Exception("Unsloth is only supported on Linux")

from dataclasses import dataclass, asdict
from enum import Enum
from src.models.basellm import BaseLLM, ChatSession
from unsloth import FastLanguageModel
import jinja2
import os


try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

@dataclass
class UnslothConfig:
    model_name: str = "unsloth/gemma-2b-it-bnb-4bit"
    max_seq_length: int = 2048
    dtype: type = None # None for auto detection
    load_in_4bit: bool = True
    system_instruction: str = None


class UnslothModel(BaseLLM):
    def __init__(self, model_config:UnslothConfig=UnslothConfig()):
        super().__init__(model_name=model_config.model_name)
        config_dict = asdict(model_config)
        self.system_instruction = config_dict.pop('system_instruction')
        self.max_seq_length = config_dict['max_seq_length']
        self.dtype = config_dict['dtype']
        self.load_in_4bit = config_dict['load_in_4bit']
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**config_dict)
        FastLanguageModel.for_inference(self.model)

        # Test if the model supports system instructions
        try:
            if self.system_instruction is not None:
                _ = self.tokenizer.apply_chat_template([
                    dict(
                        role="system",
                        content=self.system_instruction
                    )
                ])
        except jinja2.exceptions.TemplateError:
            raise Exception("System instruction is not supported by the model.")

    def generate(self, prompt, **kwargs):

        input_chatml = [dict(
            role="user",
            content=prompt
        )]

        if self.system_instruction is not None:
            input_chatml.insert(0, dict(
                role="system",
                content=self.system_instruction
            ))

        input = self.tokenizer.apply_chat_template(input_chatml, return_tensors="pt", add_generation_prompt=True).to(self.model.model.device)
        outputs = self.model.generate(input, **kwargs)[0, input.shape[1]:]
        return self.tokenizer.decode(outputs, skip_special_tokens=True)
    
    def get_chat_session(self, history=[]):
        if self.system_instruction is not None:
            history.insert(0, {"role": "system", "content": self.system_instruction})
        return UnslothChatSession(self, self.tokenizer, history)
    

class UnslothChatSession(ChatSession):
    def __init__(self, model:UnslothModel, tokenizer, history=[]):
        self.model = model
        self.tokenizer = tokenizer
        self.history = history

    def chat(self, message:str):
        self.history.append(dict(
            role="user",
            content=message
        ))

        inputs = self.tokenizer.apply_chat_template(self.history, return_tensors="pt", add_generation_prompt=True).to(self.model.model.model.device)
        outputs = self.model.model.generate(inputs)[0, inputs.shape[1]:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        self.history.append(dict(
            role="assistant",
            content=response
        ))

        return response

    def get_history(self):
        return self.history