from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1  = ChatOpenAI(model='gpt-3.5-turbo')
model2  = ChatOpenAI(model='gpt-4')


prompts1 = PromptTemplate(
    template = "Genrate  short ans simple notes from the following text \n {text}",
    input_variables = ['text']  
)
prompts2 = PromptTemplate(
    template = "Genrate 5 short question ans from the following \n {text}",
    input_variables = ['text']  
)
prompts3 = PromptTemplate(
    template = "Merge the provided notes and quiz into single document \n {notes} && {quiz} ",
    input_variables = ['notes','quiz']  
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompts1 | model1 | parser,
    "quiz": prompts2 | model2 | parser
})

merge_chain = prompts3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Support Vector Machines (SVMs) are supervised learning models used for classification, regression, and outlier detection.  Developed by Vladimir Vapnik and colleagues at AT&T Bell Laboratories in the 1960s and refined in the 1990s, SVMs work by finding an optimal hyperplane that maximizes the margin between different classes in an N-dimensional space. 

Key components include:

Support Vectors: The specific data points closest to the decision boundary that define the hyperplane's position and orientation. 
Kernel Trick: A technique that transforms non-linearly separable data into a higher-dimensional space, allowing for linear separation using kernels like Radial Basis Function (RBF), polynomial, or linear kernels. 
Soft Margin: A flexible approach using a penalty parameter (C) and slack variables to allow some misclassification, preventing overfitting in complex datasets. 
In practice, SVMs are widely implemented in Python using the scikit-learn library (via classes like SVC and LinearSVC) and are particularly effective for high-dimensional data such as text classification, image recognition, and bioinformatics. """
result =  chain.invoke({'text':text})
print(result.content)