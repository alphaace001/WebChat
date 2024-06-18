import streamlit as st
import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Set
import configparser

# Load environment variables from .env file
config = configparser.ConfigParser()
config.read(".env")
api_key = config.get("GROQ", "GROQ_API_KEY")

# Function to extract URLs
def extract_urls(base_url: str) -> Set[str]:
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.RequestException as e:
        st.warning(f"Error fetching {base_url}: {e}")
        return set()

    parsed_html = BeautifulSoup(response.content, 'html.parser')
    links = parsed_html.find_all('a', href=True)
    
    base_domain = urlparse(base_url).netloc
    urls = set()
    for link in links:
        url = link['href']
        if not urlparse(url).scheme:
            url = urljoin(base_url, url)
        if urlparse(url).netloc == base_domain:  # Check if URL is within the same domain
            urls.add(url)
    
    return urls

# Function to validate URLs
def validate_and_collect_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code < 400
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
    return False

# Function to crawl website and get URLs
def crawl_website(base_url: str, max_urls: int) -> List[str]:
    urls_to_crawl = [base_url]
    seen_urls = set()
    valid_urls = set()
    base_domain = urlparse(base_url).netloc

    while urls_to_crawl and len(valid_urls) < max_urls:
        current_url = urls_to_crawl.pop(0)
        if current_url not in seen_urls:
            seen_urls.add(current_url)
            extracted_urls = extract_urls(current_url)
            for url in extracted_urls:
                if len(valid_urls) >= max_urls:
                    break
                if url not in seen_urls and validate_and_collect_url(url):
                    if urlparse(url).netloc == base_domain:
                        valid_urls.add(url)
                        urls_to_crawl.append(url)
    
    return list(valid_urls)

# Function to load documents from multiple URLs
def load_documents_from_urls(urls: List[str]) -> List:
    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Failed to load documents from {url}: {e}")
    return all_docs

# Initialize session state variables
if "vectors" not in st.session_state:
    try:
        st.session_state.embeddings = GPT4AllEmbeddings(
            model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
            gpt4all_kwarg={'allow_download': 'True'}
        )
        st.session_state.docs = []
        st.session_state.vectors = None
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")

st.title("Multi-Website Document Loader")

# Descriptive markdown in the center
st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h2>Welcome to the Multi-Website Document Loader!</h2>
        <p>This tool allows you to extract and analyze content from multiple webpages within the same domain.</p>
        <h4>Step-by-Step Guide:</h4>
        <ol style='text-align: left; display: inline-block; text-align: left;'>
            <li><strong>Enter the Base URL:</strong> Start by entering the base URL of the website you want to analyze. The tool will extract the specified number of unique URLs from the same domain and load their content. Note: Compile time will depend on number of URLs.</li>
            <li><strong>Load Documents:</strong> The extracted URLs will be used to load documents, which will be processed and stored for further analysis.</li>
            <li><strong>Input Your Query:</strong> After the documents are loaded, you can input your questions. The chatbot will provide responses based on the context of the loaded documents.</li>
        </ol>
        <p><em>Get started by entering a base URL below!</em></p>
    </div>
""", unsafe_allow_html=True)

# Input for base URL
base_url_input = st.text_input("Enter the base URL of the website:")

# Input for number of URLs to extract
num_urls_input = st.number_input("Enter the number of URLs to extract:", min_value=1, max_value=100, value=20)

# Button to start crawling and loading documents
if st.button("Extract URLs and Load Documents"):
    if base_url_input:
        st.write("Extracting URLs, please wait...")
        urls = crawl_website(base_url_input, int(num_urls_input))
        if urls:
            st.write(f"Extracted the following {len(urls)} URLs:")
            st.write(urls)
            st.write("Loading documents from the URLs, please wait...")
            st.session_state.docs = load_documents_from_urls(urls)
            st.success(f"Loaded documents from {len(urls)} URLs successfully.")
            
            st.write("Creating vector store, please wait...")
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector store created successfully.")
        else:
            st.warning("No valid URLs could be extracted.")
    else:
        st.error("Please enter a valid base URL.")

# Initialize the ChatGroq model
llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")

# Define the prompt template
prompt_template = PromptTemplate(
    template="""
    Answer the questions based on the provided context.
    Please provide a brief and accurate response.
    <context>
    {context}
    </context>
    Question: {input}
    """,
    input_variables=["context", "input"]
)

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create a retriever using the vector store
retriever = st.session_state.vectors.as_retriever() if st.session_state.vectors else None

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain) if retriever else None

# User input for the prompt
prompt_input = st.text_input("Input your query here")

# Process the input prompt
if prompt_input and retrieval_chain:
    try:
        st.write("Processing your query, please wait...")
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": prompt_input})
        response_time = time.process_time() - start_time
        st.write(f"Response time: {response_time:.2f} seconds")
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("...................................")
    except Exception as e:
        st.error(f"Failed to retrieve answer: {e}")
