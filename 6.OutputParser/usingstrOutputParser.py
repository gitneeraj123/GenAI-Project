from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel, EmailStr,Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# open source me bahut saare model structur output support nhi krte he 
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)


# 1st promt : Detail report of topic 
template1 = PromptTemplate(
    template = "Write a detail report on the {topic} ",
    input_variables=['topic']
)

# 2nd prompt : summary of the report
template2 = PromptTemplate(
    template = "Write a 5 line summary of the following text. /n {text}",
    input_variables=['text']
)

# output parser
parser = StrOutputParser()

chains = template1 | model | parser | template2 | model | parser 

result = chains.invoke({'topic':'Black hole'})

print("Summary : ",result)

