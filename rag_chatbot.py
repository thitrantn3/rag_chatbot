import streamlit as st
import os
import time
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

# https://zapier.com/blog/perplexity-vs-chatgpt/

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🦙 RAG Chatbot with Ollama")

# Source selection
source_type = st.radio("Select document source:", ("PDF", "Website URL"))

docs = None

if source_type == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        # Save uploaded file
        pdf_path = os.path.join("uploaded.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and split document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
elif source_type == "Website URL":

    # File uploader
    url = st.text_input("Enter website URL:", placeholder="https://example.com")

    if url:
        # Load document
        loader = WebBaseLoader(url)
        docs = loader.load()

#If the documents are loaded, proceed
if docs:
    #Split document
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo",   # or "gpt-3.5-turbo", "gpt-5", etc.
            chunk_size=500,
            chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # Embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    retriever = vectorstore.as_retriever()

    # Prompt and LLM
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise.

    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = OllamaLLM(model=LLM_MODEL)
    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Display the user's question immediately
        st.chat_message("user").write(user_input)
        start_time = time.time()
        with st.spinner("Generating answer..."):
            try:
                answer = rag_chain.invoke(user_input)
            except Exception as e:
                answer = f"Error: {e}"
            elapsed = time.time() - start_time
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        st.info(f"⏱️ Response time: {elapsed:.2f} seconds")

# else document is not loaded
else:
    st.info("Please input a document to get started.")