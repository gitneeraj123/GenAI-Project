from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

loader = DirectoryLoader(
    path="marketing",
    glob = "*.pdf",
    loader_cls=PyPDFLoader
)
# docs = loader.load() 
docs = loader.lazy_load() 

# print(len(docs))
# print(docs[0].page_content)

for doc in docs:
    print(doc.metadata)

