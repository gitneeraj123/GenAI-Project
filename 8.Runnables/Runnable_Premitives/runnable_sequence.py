from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
load_dotenv()
# chain 1
prompt1 = PromptTemplate(
    template = "write a joke about {topic}",
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

# chain2
prompt2 = PromptTemplate( 
    template = "explain the following joke {response}",
    input_variables = ['response']
)

# chaining chain1 and chain2 together
chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
result = chain.invoke({'topic':'India'}) 
print(result)
