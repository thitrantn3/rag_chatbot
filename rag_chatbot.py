"""
A complete, runnable example of a Retrieval-Augmented Generation (RAG) chatbot.

This script demonstrates the core components of a RAG system using:
- LangChain for orchestrating the pipeline.
- pypdf to load and parse a PDF document.
- ChromaDB as a local, in-memory vector store for document embeddings.
- HuggingFaceEmbeddings for converting text to numerical vectors.

The RAG pipeline works as follows:
1. Load a PDF document from a specified path.
2. Split the document into smaller, manageable chunks.
3. Create numerical representations (embeddings) of these chunks.
4. Store the embeddings in a vector database (ChromaDB) for efficient retrieval.
5. When a user asks a question, retrieve the most relevant chunks from the database.
6. Pass the retrieved chunks along with the user's question to a large language model (LLM).
7. The LLM uses the provided context to generate a grounded and accurate answer.
"""
import os
import getpass
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI

# -------------------- Configuration --------------------
# Set your LLM API key. This example uses a placeholder for OpenAI.
# In a real application, you should handle this more securely (e.g., environment variables).
# export OPENAI_API_KEY="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"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please provide your OpenAI API key:")

# Define the path to your document.
# NOTE: You must have a PDF file at this location for the script to run.
# Example: Create a file named 'example.pdf' in the same directory.
DOCUMENT_PATH = " prompt_chaining.pdf"
CHROMA_DB_PATH = "./chroma_db"

# -------------------- Main RAG Logic --------------------
def create_rag_chatbot():
    """
    Sets up and runs the RAG chatbot.
    """
    print("Step 1: Loading and splitting document...")
    if not os.path.exists(DOCUMENT_PATH):
        print(f"Error: Document not found at '{DOCUMENT_PATH}'. Please place a PDF file there and try again.")
        return

    # Load the document
    loader = PyPDFLoader(DOCUMENT_PATH)
    docs = loader.load()

    # Split the document into chunks to manage context size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)

    print(f"Document split into {len(splits)} chunks.")

    print("Step 2: Creating embeddings and vector store...")
    # Use a Hugging Face model to create embeddings
    # run huggingface-cli login -> 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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a vector store from the document splits
    # This will also persist the data to disk, so it doesn't have to be re-embedded
    # on every run.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print("Vector store created and saved.")

    # Create a retriever to get relevant documents
    retriever = vectorstore.as_retriever()

    print("Step 3: Creating the RAG chain...")
    # Define the prompt template for the LLM
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise.

    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Instantiate the LLM (using a placeholder for OpenAI's model)
    llm = OpenAI()

    # Build the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain initialized. You can now ask questions!")
    print("Type 'exit' to quit.")

    # Main chatbot loop
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break

        print("Searching for answer...")
        try:
            # Invoke the RAG chain with the user's question
            response = rag_chain.invoke(user_question)
            print("Answer:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    create_rag_chatbot()
