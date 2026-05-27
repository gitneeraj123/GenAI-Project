from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate(
    template = "give the summary of the following.\n {topic}",
    input_variables=['topic']
)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

loader = TextLoader("wiki.txt")
docs = loader.load()

chain = prompt | model | parser 

result  = chain.invoke({"topic": docs[0].page_content})
print(result)
