from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = ["Bhopal is the capital of Madhya Pradesh", "Indore is the largest city in Madhya Pradesh", "Madhya Pradesh is a state in central India"]

result = embedding.embed_documents(documents)   

print(result)
