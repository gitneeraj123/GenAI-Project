from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence
load_dotenv()
prompt1 = PromptTemplate(
    template = "Generate the tweet about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate( 
    template = "Generate the LinkedIn post about {topic}",
    input_variables = ['topic']
)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    # repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# chaining chain1 and chain2 together
ParrallelChain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})
result = ParrallelChain.invoke({'topic':'AI'}) 
print(result)

