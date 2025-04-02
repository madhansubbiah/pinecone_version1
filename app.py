import sys
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

# Initialize embeddings and vector store using in-memory storage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(collection_name="example_collection", embedding_function=embeddings, persist_directory=None)

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

        try:
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
                st.error(f"API call failed with status code: {response.status_code}")
                return "Error: API call failed."
        except Exception as e:
            st.error(f"An error occurred while calling the API: {e}")
            return ""

def handle_not_satisfied(query):
    st.session_state.satisfaction = "No"
    st.write("You clicked 'Not Satisfied'. Invoking LLM to find more detailed answers...")

    # Debugging statement to check the query being passed
    st.write(f"Query passed to LLM: {query}")

    prompt = f"Given the question: {query}, please provide a detailed answer based on current knowledge."
    llm = GroqLLM(api_key=groq_api_key)
    st.session_state.llm_response = llm.invoke([{"role": "user", "content": prompt}])
    
    if st.session_state.llm_response:
        st.write("ChatGroq Response:")
        st.write(st.session_state.llm_response)
    else:
        st.write("No response was obtained from the LLM.")

# Streamlit application
st.title("Chroma and ChatGroq Query Interface")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an option", ["Data Ingestion to Chroma", "View Documents & Clear DB", "Search in Chroma DB/LLM"])

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = []
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ''
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'satisfaction' not in st.session_state:
    st.session_state.satisfaction = None

# Reset session state when navigating to Search in Chroma DB/LLM
if app_mode == "Search in Chroma DB/LLM":
    st.session_state.query = ''
    st.session_state.results = []
    st.session_state.llm_response = ''
    st.session_state.satisfaction = None  # Resetting satisfaction state

if app_mode == "Data Ingestion to Chroma":
    st.markdown("## Enter Raw Text or Upload Documents")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose a CSV or TXT file to upload.", type=["csv", "txt"])
    with col2:
        free_text_input = st.text_area("Or enter free text to store in Chroma DB:", placeholder="Type your text here...")

    if st.button("Store Document"):
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                for index, row in df.iterrows():
                    content = row['content'] if 'content' in row else str(row)
                    vector_store.add_documents([Document(page_content=content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                st.success("CSV content has been successfully stored in Chroma DB.")
            elif uploaded_file.type == "text/plain":
                text_content = uploaded_file.read().decode("utf-8")
                vector_store.add_documents([Document(page_content=text_content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                st.success("Text content has been successfully stored in Chroma DB.")
        elif free_text_input:
            vector_store.add_documents([Document(page_content=free_text_input, metadata={"source": "user_input"}, id=str(uuid4()))])
            st.success("Free text has been successfully stored in Chroma DB.")
        else:
            st.warning("Please upload a file or enter some text.")

elif app_mode == "View Documents & Clear DB":
    st.markdown("## Manage Your Documents in Chroma DB")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chroma DB"):
            try:
                vector_store.delete_collection()
                st.success("Chroma DB has been successfully cleared.")
            except Exception as e:
                st.error(f"Error clearing Chroma DB: {e}")

    with col2:
        if st.button("View Chroma Content"):
            try:
                all_documents = vector_store.similarity_search("", k=10)
                if all_documents:
                    st.subheader("Current Documents in Chroma:")
                    for doc in all_documents:
                        st.write(f"**ID:** {doc.id} | **Content:** {doc.page_content}")
                else:
                    st.info("No documents found in Chroma DB.")
            except Exception as e:
                st.error(f"Error retrieving documents: {e}")

elif app_mode == "Search in Chroma DB/LLM":
    st.markdown("## Search in Chroma DB/LLM")

    if st.session_state.satisfaction is None:
        query = st.text_input("Enter your query:", value=st.session_state.query)
        st.session_state.query = query  # Capture the query input each time it is updated
        k = st.number_input("Number of results to retrieve:", min_value=1, max_value=10, value=5)
        filter_source = st.selectbox("Select source for filtering:", ["All", "user_upload", "user_input"])

        # Buttons in the same line
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjusting column sizes proportionally
        with col1:
            if st.button("Search"):
                filter_condition = {"source": filter_source} if filter_source != "All" else None
                results = vector_store.similarity_search(query, k=k, filter=filter_condition)
                relevant_texts = [res.page_content for res in results]

                if relevant_texts:
                    scored_results = [(text, score_result(text, query)) for text in relevant_texts]
                    scored_results.sort(key=lambda x: x[1], reverse=True)
                    st.session_state.results = scored_results

                    for text, score in st.session_state.results:
                        st.write(f"**Score:** {score} | **Content:** {text}")
                else:
                    st.info("No relevant texts found in Chroma.")

        with col2:
            if st.button("Not Satisfied"):
                handle_not_satisfied(st.session_state.query)

        with col3:
            if st.button("Start New Query"):
                # Clear session state for a new query
                st.session_state.satisfaction = None
                st.session_state.query = ''
                st.session_state.results = []
                st.session_state.llm_response = ''
                st.rerun()  # Use st.rerun() for a fresh start

# Optional: Visualize the state of variables for debugging
# st.write(f"Current query: {st.session_state.query}")
# st.write(f"Current results: {st.session_state.results}")
# st.write(f"Current LLM response: {st.session_state.llm_response}")
