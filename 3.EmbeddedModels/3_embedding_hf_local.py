from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "Bhopal is the capital of Madhya Pradesh"

vector = embedding.embed_query(text)

print(str(vector))