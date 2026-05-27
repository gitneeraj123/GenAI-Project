# split text based on the length
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
docs = PyPDFLoader('Intro_To_Algo.pdf').load()

splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    separator = '\n'
)
result = splitter.split_documents(docs)
print(result)