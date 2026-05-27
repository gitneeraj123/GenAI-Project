from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template 
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])


# load chat history 
chat_history = []
with open('chat_history.txt', 'r') as file:
    chat_history.extend(file.readlines())


# create prompt 
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query':'What is the status of my refund?'
})

print(prompt)