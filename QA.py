import os
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

# Streamlit app setup
st.title("Q & A WITH COMAPNY DOCUMMENT USING GENERATIVE AI")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the CSV using LangChain's CSVLoader
    data = CSVLoader(temp_file_path, encoding="UTF-8")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Create the vector store database
    db = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embeddings
    ).from_loaders([data])

    # Input for the query
    query_input = st.text_input(
        "Enter your query:",
        "please list all your shirts with sun protection in a table in markdown and summarize each one"
    )

    # Button to submit the query
    if st.button("Submit"):
        # Initialize the OpenAI language model
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

        # Query the database
        response = db.query(query_input, llm=llm)

        # Display the response as Markdown
        st.markdown(response)

else:
    st.warning("Please upload a CSV file.")

