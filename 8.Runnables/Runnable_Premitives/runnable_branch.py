from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableBranch
load_dotenv()

prompt1 = PromptTemplate(
    template = "write a detain report on {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate( 
    template = "summarize the following text \n {text}",
    input_variables = ['text']
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


report_generation = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500,prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = report_generation | branch_chain

result = final_chain.invoke({'topic':'Russia vs Ukraine war'})
print(result)

