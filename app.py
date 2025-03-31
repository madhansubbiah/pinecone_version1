import os
import requests
import json
import pandas as pd
import streamlit as st
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
import urllib3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("API_KEY")

# Suppress warnings related to unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize HuggingFaceEmbeddings with the specified model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

class GroqLLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def invoke(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "temperature": 1,
            "max_completion_tokens": 1024,
            "top_p": 1,
            "stream": True
        }

        response = requests.post(self.url, headers=headers, json=data, stream=True, verify=False)

        if response.status_code == 200:
            collected_content = ""
            for line in response.iter_lines():
                if line:
                    line_content = line.decode('utf-8').lstrip("data: ").strip()
                    if line_content == "[DONE]":
                        break
                    try:
                        json_line = json.loads(line_content)
                        if 'choices' in json_line and json_line['choices']:
                            collected_content += json_line['choices'][0]['delta'].get('content', '')
                    except json.JSONDecodeError:
                        continue
            return collected_content
        else:
            raise Exception(f"API call failed: {response.text}")

# Streamlit application
st.title("Chroma and ChatGroq Query Interface")
st.sidebar.title("Navigation")

# Create sidebar for navigation
app_mode = st.sidebar.selectbox("Choose an option", ["Data Ingestion to Chroma", "View Documents & Clear DB", "Search in Chroma DB"])

if app_mode == "Data Ingestion to Chroma":
    # Data Ingestion Section
    st.markdown("## Enter Raw Text or Upload Documents")

    # Layout for Document Upload and Text Input
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose a CSV or TXT file to upload.", type=["csv", "txt"])
        if uploaded_file:
            st.info("Limit 200MB per file • CSV, TXT")

    with col2:
        free_text_input = st.text_area("Or enter free text to store in Chroma DB:", placeholder="Type your text here...")

    # Button to store document
    if st.button("Store Document"):
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                # Load CSV file
                df = pd.read_csv(uploaded_file)
                for index, row in df.iterrows():
                    # Assuming there's a 'content' column
                    content = row['content'] if 'content' in row else str(row)
                    vector_store.add_documents([Document(page_content=content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                st.success("CSV content has been successfully stored in Chroma DB.")
            elif uploaded_file.type == "text/plain":
                # Load TXT file
                text_content = uploaded_file.read().decode("utf-8")
                vector_store.add_documents([Document(page_content=text_content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                st.success("Text content has been successfully stored in Chroma DB.")
        elif free_text_input:
            vector_store.add_documents([Document(page_content=free_text_input, metadata={"source": "user_input"}, id=str(uuid4()))])
            st.success("Free text has been successfully stored in Chroma DB.")
        else:
            st.warning("Please upload a file or enter some text.")

elif app_mode == "View Documents & Clear DB":
    # View Documents & Clear DB Section
    st.markdown("## Manage Your Documents in Chroma DB")

    # Button to clear Chroma DB
    if st.button("Clear Chroma DB"):
        try:
            vector_store.delete_collection()
            st.success("Chroma DB has been successfully cleared.")
        except Exception as e:
            st.error(f"Error clearing Chroma DB: {e}")

    # Button to view content in Chroma DB
    if st.button("View Chroma Content"):
        try:
            # Perform a similarity search using an empty query to attempt retrieval of documents
            all_documents = vector_store.similarity_search("", k=10)

            # Check if there are documents
            if all_documents and len(all_documents) > 0:
                st.subheader("Current Documents in Chroma:")
                for doc in all_documents:
                    st.write(f"**ID:** {doc.id} | **Content:** {doc.page_content}")
            else:
                st.info("No documents found in Chroma DB.")
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")

elif app_mode == "Search in Chroma DB":
    # Search Section
    st.markdown("## Search in Chroma DB")
    
    # Query Input Section
    query = st.text_input("Enter your query:")
    k = st.number_input("Number of results to retrieve:", min_value=1, max_value=10, value=5)
    filter_source = st.selectbox("Select source for filtering:", ["All", "user_upload", "user_input"])

    if st.button("Search"):
        # Set filter condition based on user selection
        filter_condition = {"source": filter_source} if filter_source != "All" else None

        # Perform similarity search
        results = vector_store.similarity_search(query, k=k, filter=filter_condition)
        relevant_texts = [res.page_content for res in results]

        # Check if we got relevant content from Chroma
        if relevant_texts and any("football" in text for text in relevant_texts):
            relevant_text = " ".join(relevant_texts)
            st.write("Found relevant texts in Chroma:")
            st.write(relevant_text)

            # Construct a specific prompt for the LLM
            prompt = f"Based on the following content: {relevant_text}\n\nPlease answer the question: {query}."
            source_message = "Response obtained from Chroma DB."
        else:
            st.info("No relevant texts found in Chroma. Searching with LLM instead.")
            
            # If no results found in Chroma or irrelevant content, query the LLM directly
            prompt = f"Given the question: {query}, please provide a detailed answer based on current knowledge."
            source_message = "Response obtained from LLM."

        # Invoke the LLM
        llm = GroqLLM(api_key=groq_api_key)
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Display the response along with the source information
        st.write("ChatGroq Response:")
        st.write(response)
        st.text(source_message)  # Indicate where the response has come from
