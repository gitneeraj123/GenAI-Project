from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt 
load_dotenv()
import os
 
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

st.header('Reaserch Tools')

paper_input = st.selectbox("select Reaserch Paper",["select....","Attention you all need",
                                            "The Annotated Transformer",
                                            "Dynamically Fused Graph Networks for Graph-to-Graph Learning",
                                            "Diffusion Probabilistic Models for Graph Generation",])

style_input = st.selectbox("select Explanation Style",["Begginer-Freindly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("select Explanation Length",["Short","Medium","Long"])

template = load_prompt('template.json')

# fill the placeholder with user input
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.subheader('Summary')
    st.write(result.content)