import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded correctly
if not api_key:
    st.error("❌ GROQ_API_KEY is missing! Please set it in your .env file.")
    st.stop()

# Initialize ChatGroq with the specified model
llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.2)

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
        try:
            response = chain.invoke({'question': input_text})
            st.write(response)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
