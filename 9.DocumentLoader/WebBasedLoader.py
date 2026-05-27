from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["USER_AGENT"] = "Mozilla/5.0"
docs = WebBaseLoader(
    "https://www.nitt.edu/home/academics/"
).load()

# print(docs[0].page_content[:5000])

prompt = PromptTemplate(
    template="""
Answer the following question using the provided content.

Question:
{question}

Content:
{content}
""",
    input_variables=['question', 'content']
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    'question': "Give me Telephone number of NIT Trichy?",
    'content': docs[0].page_content
})

print(result)
