from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
load_dotenv()

def wordCount(text):
    return len(text.split())

# runnable_word_counter = RunnableLambda(wordCount)

prompt1 = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "explain the following joke {response}",
    input_variables=['response']
)

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

joke_generator_chain = RunnableSequence(prompt1,model,parser)

parrallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    # 'word_count':RunnableLambda(wordCount)
    'word_count':RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_generator_chain,parrallel_chain)

result = final_chain.invoke({'topic':'Life'})
print(result['joke'])
print(result['word_count'])