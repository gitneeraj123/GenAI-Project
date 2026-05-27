from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings
)
text = """
Artificial Intelligence is transforming industries.
Machine learning is a subset of AI.
Pizza is a popular Italian food.
Neural networks are inspired by the human brain.
"""

chunks = splitter.split_text(text)
print(chunks)