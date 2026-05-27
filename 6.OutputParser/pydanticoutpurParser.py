from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel, EmailStr,Field
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser, PydanticOutputParser
load_dotenv()

# open source me bahut saare model structur output support nhi krte he 
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person"),
    age: int = Field(description="The age of the person"),
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Generate the name , age , city of fictional {place} person \n {format_instruction} ",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.format(place="delhi") 
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# chain
chain = template | model | parser
result = chain.invoke({"place":"indore"})
print(result)