from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers.pydantic import ( PydanticOutputParser )
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()
model1  = ChatOpenAI(model='gpt-3.5-turbo')
parser1 = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="The sentiment of the feedback")   

parser2 = PydanticOutputParser(pydantic_object=feedback)    

prompt1  = PromptTemplate(
    template = "Classify the sentiment of following feedback text into positive, negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

# classifier chain : prompt to model to parser
classifier_chain = prompt1 | model1 | parser2

sentiment = classifier_chain.invoke({"feedback":"The product is really good and I am satisfied with the quality."}).sentiment

prompt2 = PromptTemplate(
    template = "Generate a appropriate response for the positive feedback \n {feedback} ",
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = "Generate a appropriate response for the positive feedback \n {feedback} ",
    input_variables=['feedback']
)

# branching : applying condition
branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt2 | model1 | parser1),
    (lambda x: x.sentiment=='negative', prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "Invalid sentiment")  
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback":"The product is really good and I am satisfied with the quality."})
print(result)


