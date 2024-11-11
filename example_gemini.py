from src.models.gemini import Gemini, GeminiConfig

# Initialize the Gemini model
gemini = Gemini()

# Initialize model with custom settings
model_config = GeminiConfig(
    model="gemini-1.5-flash",
    temperature=0.5,
    top_p=0.9,
    top_k=40,
    max_output_tokens=100,
    response_mime_type="text/plain",
    safety_level="none",
    system_instruction="You are a helpful assistant."
    )
gemini = Gemini(model_config=model_config)

# Generate a response
prompt = "Write a thing about yourself:"
response = gemini.generate(prompt)
print("PROMPT: ", prompt)
print("OUTPUT: ", response)

print("\n\n")

# Get a chat session
chat_session = gemini.get_chat_session()

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