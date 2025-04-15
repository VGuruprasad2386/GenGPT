import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq with the specified model
llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.2)
# st.image("xoxo.png", width=200)
# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Initialize the output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

# Streamlit UI setup
st.title("GPT Chatbot")

input_text = st.text_input("Search the topic you want:")

if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({'question': input_text})
    st.write(response)
