import random
class Naklillm:
    def __init__(self):
        print('LLM created')

    def predict(self,prompt):
        response = [
            'Delhi is the capital of India.',
            'Mumbai is the financial capital of India.',
            'Kolkata is the cultural capital of India.',
            'AI stands for Artificial Intelligence.',
            'Python is a popular programming language.'
        ]
        return {'response': random.choice(response)}
    

llm  = Naklillm()

# result = llm.predict("What is the capital of India?")

class NakliPromptTemplate:
    def __init__(self,template,input_variable):
        self.template = template
        self.input_variables = input_variable

    def format(self,input_dict):
        # unpacking the input dictionary to format the template string
        return self.template.format(**input_dict) 
    

template = NakliPromptTemplate(
    template = "write a {length} poem about the {topic}",
    input_variable = ['length','topic']
)

prompt = template.format({'topic':'moon','length':'short'}) 

result =  llm.predict(prompt)
# print(result)


# hmare pass do componenet he ab unko combine krna he ek chain
#  me taki hum prompt template se prompt generate kr ske aur usko llm me daal ke response le ske
class NaklillmChain:
    def __init__(self,llm,prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self,input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result['response']


chain = NaklillmChain(llm,prompt)
result = chain.run({'topic':'sun','length':'long'})
# print(result)

# but ye flexible nhi he ydi hme 3 component add krne he to hme chain class me changes krne padenge
#  isliye hum Runnable class ka use krenge jo ki flexible he aur easily extendable he 

# main problem is baat se a rhi he ki jo prompt template vali class he usse interact krne ka trika he format or jo llm vali class
# he usse interact krne ka trika he predict hme kisi trike se in dono class ko standardize krna padega tbhi hm flexible chain bna payenge
# solution 



