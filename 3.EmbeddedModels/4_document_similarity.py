# we have 5 documents find similarity of query using word embeddings

# for openAI 
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# load_dotenv()
# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)


# for hugging face
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
documents = [
    "Sachin Tendulkar is a former Indian cricketer",
    "Virat Kohli is the current captain of Indian cricket team",
    "M.S. Dhoni is a former Indian cricketer and captain",
    "Rohit Sharma is an Indian cricketer and opening batsman",
    "Anil Kumble is a former Indian cricketer and coach"
]

query = "Who is the current captain of Indian cricket team?"

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

# Cosine similarity
similarity_scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Get highest similarity
index, score = sorted(
    list(enumerate(similarity_scores)),
    key=lambda x: x[1],
    reverse=True
)[0]

print("Query:", query)
print("Most Similar Document:", documents[index])
print("Similarity Score:", score)