from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat  = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)  
result = chat.invoke("What is the capital of Madhya Pradesh ?")
print(result.content)