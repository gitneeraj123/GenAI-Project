# split text based on the length
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from dotenv import load_dotenv
load_dotenv()

# python code to be split
text = """  
    class Document:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata
        
        def __repr__(self):
            return f"Document(page_content={self.page_content}, metadata={self.metadata})"
        
        def length(self,DataType):
            if DataType == "page_content":
                return len(self.page_content)
            elif DataType == "metadata":
                return len(self.metadata)
            else:
                raise ValueError("Invalid DataType. Choose 'page_content' or 'metadata'.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)
result = splitter.split_text(text)
print(len(result[0]))
print(result[0])