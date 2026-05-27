from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
load_dotenv()

prompt1 = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate( 
    template = "explain the following joke {response}",
    input_variables = ['response']
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

joke_generator_chain = RunnableSequence(prompt1,model,parser)
parrallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_generator_chain,parrallel_chain)
result = final_chain.invoke({'topic':'computer'})
print(result['joke']) 
print(result['explanation'])


