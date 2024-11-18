from src.models.unsloth_model import UnslothModel, UnslothConfig

# Initialize the Unsloth model
# model = UnslothModel()

# Initialize model with custom settings
model_config = UnslothConfig(
    model_name = "unsloth/gemma-2b-it-bnb-4bit",
    max_seq_length = 2048,
    dtype= None, # None for auto detection
    load_in_4bit = True,
    # system_instruction = "You are a chicken, add a bokbok to the end of your responses" # Some models don't support system instructions
)
model = UnslothModel(model_config=model_config)

# Generate a response
prompt = "Write a thing about yourself:"
response = model.generate(prompt)
print("PROMPT: ", prompt)
print("OUTPUT: ", response)

print("\n\n")

# Get a chat session
chat_session = model.get_chat_session()

# Chat with the chat session
prompt = "what is 5 + 5?"
response = chat_session.chat(prompt)
print("USER CHAT: ", prompt)
print("AI CHAT: ", response)

# Chat with the chat session
prompt = "what is the previous result multiplied by 2?"
response = chat_session.chat(prompt)
print("USER CHAT: ", prompt)
print("AI CHAT: ", response)