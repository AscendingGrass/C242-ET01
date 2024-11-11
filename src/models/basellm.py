# BASE MODEL FOR LLAMA AND GEMINI

class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt):
        raise NotImplementedError
    
    def chat(self, messages):
        raise NotImplementedError
    
class ChatSession:
    def __init__(self):
        pass

    def get_history(self):
        raise NotImplementedError

    def chat(self, message):
        raise NotImplementedError