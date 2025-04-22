import streamlit as st
import openai
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv
load_dotenv()


#langsmith tracing

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]="True"


#create chatprompt template
Chat_prompt=ChatPromptTemplate.from_messages([
    ("system","Answer the question based on your ability"),
    ("user","question: {question}")
])

def generate_response (question,api_key,llm,temperature,max_tokens):
    groq_api=api_key
    llm=ChatGroq(model=llm,api_key=groq_api)
    parser=StrOutputParser()
    chain=Chat_prompt | llm | parser
    answer=chain.invoke({"question":question})
    return answer


    #streamlit framework

st.title("enhanced QnA chatBot with LLM")
# Sidebar for settings 
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your api key",type="password")

#Drop Down for selecting models
llm=st.sidebar.selectbox("Select an LLM models",["gemma2-9b-it","Allam-2-7b","Llama3-8b-8192"])

#adjust response max tokens and tempreature 
tempreature=st.sidebar.slider("tempreature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

#main interface for user input
st.write("Ask your quories")
user_input=st.text_input("you:")


#generate response

if user_input:
    response=generate_response(user_input,api_key,llm,tempreature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the correct quories")






