from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Create a langchain Documents for IPL players

doc1 = Document(
    page_content="Virat Kohli is an Indian cricketer and former captain of the Indian national team.",
    metadata={"player": "Virat Kohli", "team": "India"}
)

doc2 = Document(   
    page_content="MS Dhoni is an Indian cricketer and former captain of the Indian national team.",
    metadata={"player": "MS Dhoni", "team": "India"}
)

doc3 = Document(
    page_content="AB de Villiers is a South African cricketer and former captain of the South African national team.",
    metadata={"player": "AB de Villiers", "team": "South Africa"}
)
doc4 = Document(
    page_content="Chris Gayle is a West Indian cricketer and one of the most destructive batsmen in the history of T20 cricket.",
    metadata={"player": "Chris Gayle", "team": "West Indies"}
)
doc5 = Document(
    page_content="Rohit Sharma is an Indian cricketer and the current captain of the Indian national team.",
    metadata={"player": "Rohit Sharma", "team": "India"}
)


# Create a list of documents
docs = [doc1, doc2, doc3, doc4, doc5]


vector_store = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    persist_directory="chroma_db",
    collection_name="ipl_players"
)

print(vector_store)

result =  vector_store.get(include=["documents", "embeddings", "metadatas"])

print(result)

# similarity search 
query = "Who is the captain of the Indian national team?"
similar_docs = vector_store.similarity_search(query, k=2)
print(similar_docs)


# search with similarity score 
similar_docs_with_score = vector_store.similarity_search_with_score(query, k=2)
print(similar_docs_with_score)


# meta data filtering 
similar_docs_with_metadata = vector_store.similarity_search(query, k=2, filter={"team": "India"})

# add the document with the same id to update the document
docs = [
    Document( page_content="Virat Kohli plays for RCB", metadata={"player": "Virat Kohli"})
]
ids = ["virat_1"]
vector_store.add_documents( documents=docs, ids=ids)

# update document
update_doc = Document(
    page_content="Virat Kohli is an Indian cricketer and former captain.",
    metadata={"player": "Virat Kohli", "team": "India"}
)

vector_store.update_documents(
    ids=["virat_1"],
    documents=[update_doc]
)

