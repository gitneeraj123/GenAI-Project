from langchain_openai import OpenAI
from dotenv import load_dotenv

#load environment variables from .env file
load_dotenv()

llm  = OpenAI(model="gpt-3.5-turbo", temperature=0.9)
result =   llm.invoke("What is the capital of France?")
print(result)


