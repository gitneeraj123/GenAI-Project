from abc import ABC, abstractmethod

#1. Runnable is an abstract base class that defines the interface for all runnables.
class Runnable(ABC):
    @abstractmethod
    def invoke(self,input_dict):
        pass

#2. Now we will make our LLM class and PromptTemplate class inherit from Runnable and implement the invoke method.
import random
class Naklillm(Runnable):
    def __init__(self):
        print('LLM created')

    # implement the invoke method to return a random response from the list of responses
    def invoke(self,input_dict):
        response = [
            'Delhi is the capital of India.',
            'Mumbai is the financial capital of India.',
            'Kolkata is the cultural capital of India.',
            'AI stands for Artificial Intelligence.',
            'Python is a popular programming language.'
        ]
        return {'response': random.choice(response)}

    # isko nhi htana chahiye kyuki purane AI engineer he vo abhi tk apne code me predict ko use kr rhe he 
    # ydi hta diya to unke code ftne lg jayenge to ideal hme ek message print kr dena chahiye ki ye method depriciate ho gyi he 
    # def predict(self,prompt):
    #     response = [
    #         'Delhi is the capital of India.',
    #         'Mumbai is the financial capital of India.',
    #         'Kolkata is the cultural capital of India.',
    #         'AI stands for Artificial Intelligence.',
    #         'Python is a popular programming language.'
    #     ]
    #     return {'response': random.choice(response)}

# 3.Prompt class
class NakliPromptTemplate(Runnable):
    def __init__(self,template,input_variable):
        self.template = template
        self.input_variables = input_variable

    def invoke(self,input_dict):
        # unpacking the input dictionary to format the template string
        return self.template.format(**input_dict)

    def format(self,input_dict):
        # unpacking the input dictionary to format the template string
        return self.template.format(**input_dict) 

# 4.class: hme output se string nikal ke dena output: {'response': 'Delhi is the capital of India.'} 
class NakliStrOutParser(Runnable):
    def __init__(self):
        pass

    def invoke(self,input_data):
        return input_data['response']


# 5. class Purpose : 2 ya 2 se jyada components ko chain together kr pao 
class RunnableConnector(Runnable):
    def __init__(self,runnable_list):
        self.runnable_list = runnable_list

    def invoke(self,input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

# demo 
# llm = Naklillm()
# template = NakliPromptTemplate(
#     template = "write a {length} poem about the {topic}",
#     input_variable = ['length','topic']
# )
# parser = NakliStrOutParser()
# chain = RunnableConnector([template,llm,parser]) 
# result = chain.invoke({'topic':'moon','length':'short'})
# print(result)


# example: application chain1(joke generate krega) ko hm chain2(us joke ka explanation dega) se connect krenge 
template1 = NakliPromptTemplate(
    template = "write a Joke about {topic}",
    input_variable=['topic']
)
template2 = NakliPromptTemplate(
    template  = "explain the following joke {response}",
    input_variable= ['response']
)
llm = Naklillm()
parser = NakliStrOutParser()
chain1 = RunnableConnector([template1,llm])
chain2 = RunnableConnector([template2,llm,parser])
# hm manually invoke nhi krenge like chain1.invoke({'topic':'computer'}) and uske output ko chain2 me daal ke invoke krenge
final_chain = RunnableConnector([chain1,chain2])
result = final_chain.invoke({'topic':'cricket'})
print(result)
