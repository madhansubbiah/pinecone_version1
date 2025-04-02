import os
import requests
import json
import pandas as pd
import streamlit as st
from uuid import uuid4
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.schema import Document
import urllib3
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec  # Updated import for Pinecone initialization

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Suppress warnings related to unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up proxy for requests (assuming HTTP_PROXY and HTTPS_PROXY environment variables are set)
proxy = {
    'http': 'http://webproxy.merck.com:8080',
    'https': 'http://webproxy.merck.com:8080',
}

# Custom function to disable SSL certificate verification for Pinecone API requests
class NoVerifySession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False  # Disable SSL verification

# Initialize Pinecone with custom session that disables SSL verification
try:
    # Creating a custom session to disable SSL verification
    session = NoVerifySession()
    
    # Pass the session to the Pinecone client
    pc = Pinecone(api_key=pinecone_api_key, session=session)  # Corrected Pinecone initialization with custom session
    st.success("Pinecone initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Specify your index name (ensure it conforms to Pinecone naming conventions)
index_name = "example-index"  # Use lower case and hyphens

# Initialize embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize vector_store as None initially
vector_store = None

# Check if the index exists; create it if it doesn't
try:
    # Bypass SSL verification and use proxy for Pinecone API request
    existing_indexes = pc.list_indexes().names()  # Updated method to list indexes with proxy and no SSL verification
    st.write(f"Existing indexes: {existing_indexes}")  # Debugging output
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',  # Cloud provider (e.g., aws)
                region='us-west-2'  # Region (e.g., us-west-2)
            )
        )
        st.success(f"Index '{index_name}' created successfully.")
    else:
        st.success(f"Index '{index_name}' already exists.")
except requests.exceptions.SSLError as ssl_error:
    st.error(f"SSL certificate verification failed: {ssl_error}")
except Exception as e:
    st.error(f"Error while checking or creating index: {e}")
    st.stop()

# Initialize LangchainPinecone vector store correctly
try:
    vector_store = LangchainPinecone(index_name=index_name, embedding_function=embeddings_model.embed_query)
    st.success("Vector store initialized successfully.")
except Exception as e:
    st.error(f"Error initializing vector store: {e}")  # Handle initialization error
    st.stop()

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
            #response = requests.post(self.url, headers=headers, json=data, stream=True, verify=False, proxies=proxy)
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

    prompt = f"Given the question: {query}, please provide a detailed answer based on current knowledge."
    llm = GroqLLM(api_key=groq_api_key)
    st.session_state.llm_response = llm.invoke([{"role": "user", "content": prompt}])
    
    if st.session_state.llm_response:
        st.write("ChatGroq Response:")
        st.write(st.session_state.llm_response)
    else:
        st.write("No response was obtained from the LLM.")

# Streamlit application
st.title("Pinecone and ChatGroq Query Interface")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an option", ["Data Ingestion to Pinecone", "View Documents & Clear Index", "Search in Pinecone/LLM"])

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = []
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = ''
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'satisfaction' not in st.session_state:
    st.session_state.satisfaction = None

# Reset session state when navigating to Search in Pinecone/LLM
if app_mode == "Search in Pinecone/LLM":
    st.session_state.query = ''
    st.session_state.results = []
    st.session_state.llm_response = ''
    st.session_state.satisfaction = None  # Resetting satisfaction state

if app_mode == "Data Ingestion to Pinecone":
    st.markdown("## Enter Raw Text or Upload Documents")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose a CSV or TXT file to upload.", type=["csv", "txt"])
    with col2:
        free_text_input = st.text_area("Or enter free text to store in Pinecone:", placeholder="Type your text here...")

    if st.button("Store Document"):
        if vector_store is None:
            st.error("Vector store is not initialized. Please check the settings.")
        else:
            if uploaded_file is not None:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    for index, row in df.iterrows():
                        content = row['content'] if 'content' in row else str(row)
                        vector_store.add_documents([Document(page_content=content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                    st.success("CSV content has been successfully stored in Pinecone.")
                elif uploaded_file.type == "text/plain":
                    text_content = uploaded_file.read().decode("utf-8")
                    vector_store.add_documents([Document(page_content=text_content, metadata={"source": "user_upload"}, id=str(uuid4()))])
                    st.success("Text content has been successfully stored in Pinecone.")
            elif free_text_input:
                vector_store.add_documents([Document(page_content=free_text_input, metadata={"source": "user_input"}, id=str(uuid4()))])
                st.success("Free text has been successfully stored in Pinecone.")
            else:
                st.warning("Please upload a file or enter some text.")

elif app_mode == "View Documents & Clear Index":
    st.markdown("## Manage Your Documents in Pinecone Index")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Pinecone Index"):
            try:
                pc.delete_index(index_name)  # Adjust as necessary; consider using clear or delete functions
                st.success("Pinecone Index has been successfully cleared.")
            except Exception as e:
                st.error(f"Error clearing Pinecone Index: {e}")

    with col2:
        if st.button("View Pinecone Content"):
            if vector_store is None:
                st.error("Vector store is not initialized. Please check the settings.")
            else:
                try:
                    all_documents = vector_store.similarity_search("", k=10)
                    if all_documents:
                        st.subheader("Current Documents in Pinecone:")
                        for doc in all_documents:
                            st.write(f"**ID:** {doc.id} | **Content:** {doc.page_content}")
                    else:
                        st.info("No documents found in Pinecone Index.")
                except Exception as e:
                    st.error(f"Error retrieving documents: {e}")

elif app_mode == "Search in Pinecone/LLM":
    st.markdown("## Search in Pinecone/LLM")

    if st.session_state.satisfaction is None:
        query = st.text_input("Enter your query:", value=st.session_state.query)
        st.session_state.query = query  # Capture the query input each time it is updated
