from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate the 5 important pillars of {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate( 
    template = 'Generate the 5 pointer summary for following text {topic} \n',
    input_variables = ['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic":'Artificial Intelligence'})
print(result.content)