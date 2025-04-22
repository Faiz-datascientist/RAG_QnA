import streamlit as st
import os 
from langchain_groq.chat_models import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS,Chroma
from langchain_text_splitters import TextSplitter,RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

#load enviroment 
from dotenv import load_dotenv
load_dotenv()

#Api key 
os.environ["Groq_API"]=os.getenv("Groq_API")
os.environ["HF_KEY"]=os.getenv("HF_KEY")

Api_key=os.getenv("Groq_API")
llm=ChatGroq(model="Llama3-8b-8192",api_key=Api_key)
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


prompts=ChatPromptTemplate.from_messages([
    ("system","Answer the quories by your ability with {context} and answer consisely"),
    ("user","{input}")
])


#create vector store in function manner

# def VectorStore_embeddings():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         st.session_state.loader=PyPDFDirectoryLoader("Attention.pdf") #data ingestion
#         st.session_state.docs=st.session_state.loader.load()
#         st.session_state.text_split=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
#         st.session_state.final_docs=st.session_state.text_split.split_documents(st.session_state.docs[:50])
#         st.session_state.vector=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("\research_papers") ## Data Ingestion step
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.write("RAG documented with Llama3")
#User input    
user_prompt=st.text_input("Enter your quories from research paper")

if st.button("Document embedding"):
    create_vector_embedding()
    st.write("Vector database is ready!")


import time
if user_prompt:
    combine_document=create_stuff_documents_chain(llm,prompts)
    retriever=st.session_state.vectors.as_retriever()
    create_retriever=create_retrieval_chain(retriever,combine_document)
    start=time.process_time()
    response=create_retriever.invoke({"input":user_prompt})
    print(f"Response time :{time.process_time()-start}")


    st.write(response["answer"])


    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

#stream lit expander 


# with st.expander("Document Expander"):
#     for i,doc in enumerate(response['context']):
#         st.write(doc.page_content)




