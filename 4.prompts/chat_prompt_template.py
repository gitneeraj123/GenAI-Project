from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# here different method than for creating chat template than prompt template(uses SystemMessage, HumanMessage, AIMessage instead of string template)
chat_template = ChatPromptTemplate([
   ('system', "You are a helpful {domain} expert."),
    ('human', "Explain me in simple terms. what is {topic}?")
])


prompt= chat_template.invoke({
    "domain":"AI",
    "topic":"machine learning"  
})

print(prompt)