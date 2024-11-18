import google.generativeai as genai
from google.generativeai.types import (
    HarmCategory,
    HarmBlockThreshold,
)
from dataclasses import dataclass, asdict
from enum import Enum
from src.models.basellm import BaseLLM, ChatSession
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

class SafetyLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class GeminiConfig:
    model: str = "gemini-1.5-flash"
    temperature: float = 1
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 2056
    response_mime_type: str = "text/plain"
    safety_level: SafetyLevel = SafetyLevel.NONE
    system_instruction: str = None

def get_safety_settings(safety_level: SafetyLevel):
    if safety_level == SafetyLevel.NONE:
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT : HarmBlockThreshold.BLOCK_NONE
        }
    elif safety_level == SafetyLevel.LOW:
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT : HarmBlockThreshold.BLOCK_ONLY_HIGH
        }
    elif safety_level == SafetyLevel.MEDIUM:
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        }
    elif safety_level == SafetyLevel.HIGH:
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT : HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        }

class Gemini(BaseLLM):
    def __init__(self, model_config:GeminiConfig=GeminiConfig()):
        
        """
        Initializes a Gemini model object.

        Args:
            model_config (GeminiConfig, optional): A GeminiConfig object. Defaults to GeminiConfig().

        Attributes:
            model_name (str): The model name.
            model (genai.LanguageModel): The underlying language model object from the GenAI SDK.

        Example:
            >>> model = Gemini()
            >>> print(model.generate("Write a thing about yourself:"))
            >>> # "I'm an AI!"
        """

        super().__init__(model_config.model)
        self.set_model_from_config(model_config)
        
    def set_model_from_config(self, model_config:GeminiConfig):
        self.model_name = model_config.model
        generation_config = asdict(model_config)
        model_name = generation_config.pop('model')
        safety_level = generation_config.pop('safety_level')
        system_instruction = generation_config.pop('system_instruction')
        
        settings = dict(
            model_name = model_name,
            generation_config = generation_config,
            safety_settings = get_safety_settings(safety_level)
        )

        if system_instruction:
            settings['system_instruction'] = system_instruction

        self.model = genai.GenerativeModel(**settings)

    def generate(self, prompt:str, return_response:bool=False, **kwargs):
        response = self.model.generate_content(prompt, **kwargs)
        if return_response:
            return response
        return response.text
    
    def get_chat_session(self, history=[]) -> ChatSession:
        """
        Creates a new chat session with the given history.

        Args:
            history (list, optional): List of messages to initialize the chat session with. Defaults to [].

        Returns:
            ChatSession: A new GeminiChatSession object.

        Example:
            >>> model = Gemini()
            >>> chat_session = model.get_chat_session()
            >>> print(chat_session.chat("Hello, how are you?"))
            >>> # "I'm doing well, thank you!"
            >>> print(chat_session.chat("Can you tell me about this thing?"))
            >>> # "Yes, it's a thing."
        """
        return GeminiChatSession(model=self, history=history)
    
class GeminiChatSession(ChatSession):
    def __init__(self, model: Gemini, history=[]):
        '''
        Initializes a GeminiChatSession object. Use the model's get_chat_session method instead.
        
        Args:
            model (Gemini): Gemini model object.
            history (list, optional): List of messages to initialize the chat session with. Defaults to [].
        
        Attributes:
            model (Gemini): The Gemini model object.
            session (genai.ChatSession): The underlying chat session object from the GenAI SDK.

        Example:
            >>> model = Gemini()
            >>> chat_session = GeminiChatSession(model)
            >>> print(chat_session.chat("Hello, how are you?"))
            >>> # "I'm doing well, thank you!"
            >>> print(chat_session.chat("Can you tell me about this thing?"))
            >>> # "Yes, it's a thing."
        '''
        super().__init__()
        self.model = model
        self.session = model.model.start_chat(history=history)

    def get_history(self):
        return self.session.history

    def chat(self, message:str, return_response:bool=False):
        response = self.session.send_message(message)
        if return_response:
            return response
        return response.text