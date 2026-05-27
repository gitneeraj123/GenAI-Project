from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load your environment variables (.env file)
load_dotenv()

# Ensure you are using a model supported by the free Inference API
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)
 
# Test the chatbot
result = model.invoke("What is the capital of France?")
print(result.content)