#!/usr/bin/env python3
"""
Knowledge Extraction from Engineering PDFs: RAG Approach
A complete implementation for extracting knowledge from engineering PDFs using 
Retrieval-Augmented Generation (RAG)
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import warnings
import pickle
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directories for storing files
os.makedirs('pdf_data', exist_ok=True)
os.makedirs('vector_db', exist_ok=True)

# Check and install required packages if not already installed
def check_and_install_dependencies():
    """Check and install required packages if not already installed"""
    try:
        import pkg_resources
        required_packages = [
            'PyPDF2', 'langchain', 'sentence-transformers', 'chromadb', 
            'torch', 'transformers', 'openai', 'pypdf', 'faiss-cpu', 'tiktoken'
        ]
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in required_packages if pkg.lower() not in installed]
        
        if missing:
            print(f"Installing missing packages: {missing}")
            import subprocess
            subprocess.check_call(['pip', 'install'] + missing)
            print("Packages installed successfully!")
        else:
            print("All required packages are already installed!")
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        print("Please install the required packages manually: PyPDF2, langchain, sentence-transformers, chromadb, torch, transformers, openai, pypdf, faiss-cpu, tiktoken")


# Now import the rest of the dependencies
def import_dependencies():
    """Import all required dependencies"""
    global PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, Chroma
    global RetrievalQA, OpenAI, SentenceTransformer, chromadb, openai
    
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        from sentence_transformers import SentenceTransformer
        import chromadb
        import openai
        print("All dependencies imported successfully!")
        return True
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        return False


def setup_api_key():
    """Set up OpenAI API key"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print("API key set!")


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        tuple: (full_text, pages) where full_text is a string and pages is a list of document objects
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Extract text from each page
    full_text = ""
    for page in pages:
        full_text += page.page_content + "\n\n"
    
    # Basic cleaning
    # Remove excessive whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    # Remove page numbers
    full_text = re.sub(r'\b\d{1,3}\b', '', full_text)
    
    return full_text, pages


def split_text(documents):
    """
    Split text into chunks for better processing
    
    Args:
        documents (list): List of document objects
        
    Returns:
        list: List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def initialize_embedding_model():
    """
    Initialize the embedding model for vectorizing text
    
    Returns:
        HuggingFaceEmbeddings: Embedding model
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Good for technical content
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    print(f"Embedding model initialized! Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return embeddings


def create_vector_store(chunks, embeddings):
    """
    Create a vector store from document chunks
    
    Args:
        chunks (list): List of document chunks
        embeddings (HuggingFaceEmbeddings): Embedding model
        
    Returns:
        Chroma: Vector store
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )
    
    # Persist the database
    vectordb.persist()
    print("Vector store created and persisted!")
    return vectordb


def process_multiple_pdfs(pdf_directory, embeddings):
    """
    Process multiple PDF files from a directory
    
    Args:
        pdf_directory (str): Path to directory containing PDF files
        embeddings (HuggingFaceEmbeddings): Embedding model
        
    Returns:
        Chroma: Vector store or None if no PDFs are found
    """
    all_chunks = []
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return None
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            _, pdf_pages = extract_text_from_pdf(pdf_path)
            # Add file name to metadata
            for page in pdf_pages:
                page.metadata["source"] = pdf_file
            
            chunks = split_text(pdf_pages)
            all_chunks.extend(chunks)
            print(f"Processed {pdf_file}: {len(chunks)} chunks extracted")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    # Create vector store
    if all_chunks:
        vectordb = create_vector_store(all_chunks, embeddings)
        print(f"Vector database created with {len(all_chunks)} chunks from {len(pdf_files)} files")
        return vectordb
    else:
        print("No chunks extracted from PDFs")
        return None


def setup_rag_system(vector_store):
    """
    Set up the Retrieval-Augmented Generation system
    
    Args:
        vector_store (Chroma): Vector store
        
    Returns:
        RetrievalQA: QA chain
    """
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 matches
    )
    
    # Initialize the language model
    llm = OpenAI(temperature=0.0)  # Low temperature for factual responses
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Other options: map_reduce, refine
        retriever=retriever,
        return_source_documents=True
    )
    
    print("RAG system set up!")
    return qa_chain


def query_engineering_knowledge(qa_chain, query):
    """
    Query the engineering knowledge base
    
    Args:
        qa_chain (RetrievalQA): QA chain
        query (str): Query string
        
    Returns:
        dict: Result dictionary containing answer and source documents
    """
    print(f"\nQuerying: {query}")
    print("-" * 80)
    
    result = qa_chain({"query": query})
    
    # Display the answer
    print("\nAnswer:")
    print("-" * 80)
    print(result["result"])
    print("-" * 80)
    
    # Display the source documents
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\nSource {i+1}:")
        print(f"Content: {doc.page_content[:150]}...")  # Show first 150 chars
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    return result


def query_with_openai_directly(vector_store, query, model="gpt-3.5-turbo"):
    """
    Alternative approach using OpenAI API directly with retrieved context
    
    Args:
        vector_store (Chroma): Vector store
        query (str): Query string
        model (str): OpenAI model to use
        
    Returns:
        tuple: (answer, docs) where answer is a string and docs is a list of documents
    """
    print(f"\nQuerying with OpenAI directly: {query}")
    print("-" * 80)
    
    # Get relevant documents
    docs = vector_store.similarity_search(query, k=3)
    
    # Extract context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt with context
    prompt = f"""
    You are an engineering knowledge assistant. Use the following extracted parts of engineering documents to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    YOUR ANSWER:
    """
    
    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an engineering knowledge assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    answer = response.choices[0].message.content
    
    # Display the answer
    print("\nAnswer:")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    
    # Display the source documents
    print("\nSources:")
    for i, doc in enumerate(docs):
        print(f"\nSource {i+1}:")
        print(f"Content: {doc.page_content[:150]}...")  # Show first 150 chars
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    return answer, docs


def evaluate_rag_system(qa_chain, test_questions, ground_truth=None):
    """
    Evaluate the RAG system using test questions
    
    Args:
        qa_chain (RetrievalQA): QA chain
        test_questions (list): List of test questions
        ground_truth (list): List of ground truth answers (optional)
        
    Returns:
        pd.DataFrame: DataFrame containing evaluation results
    """
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"Testing question {i+1}: {question}")
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = [doc.page_content[:100] + "..." for doc in result["source_documents"]]
        
        results.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })
        
        # If ground truth is provided, compare
        if ground_truth and i < len(ground_truth):
            print(f"Ground truth: {ground_truth[i]}")
            print(f"Model answer: {answer}")
            # Here you could add automatic metrics
        
        print("-" * 80)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df


def save_project_state(vectordb, filename="engineering_rag_state"):
    """
    Save important project variables
    
    Args:
        vectordb (Chroma): Vector store
        filename (str): Filename to save to
    """
    # We can only save certain components
    # Vector DB is already persisted on disk
    
    # Save any other variables if needed
    project_info = {
        "creation_date": pd.Timestamp.now(),
        "num_documents": vectordb._collection.count()
    }
    
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(project_info, f)
    
    print(f"Project state saved to {filename}.pkl")


def load_vector_db(embeddings):
    """
    Load the vector database
    
    Args:
        embeddings (HuggingFaceEmbeddings): Embedding model
        
    Returns:
        Chroma: Vector store or None if not found
    """
    if os.path.exists("./vector_db"):
        vectordb = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
        print(f"Loaded vector database with {vectordb._collection.count()} documents")
        return vectordb
    else:
        print("No vector database found at ./vector_db")
        return None


def interactive_mode(vectordb):
    """
    Interactive mode for querying the knowledge base
    
    Args:
        vectordb (Chroma): Vector store
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE ENGINEERING KNOWLEDGE BASE")
    print("=" * 80)
    print("Type 'exit' to quit")
    print("Type 'openai' to switch to direct OpenAI querying")
    print("Type 'langchain' to switch to LangChain QA")
    print("=" * 80)
    
    # Set up RAG system
    qa_chain = setup_rag_system(vectordb)
    
    # Default mode
    mode = "langchain"
    
    while True:
        query = input("\nEnter your engineering question: ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        elif query.lower() == 'openai':
            mode = "openai"
            print("Switched to direct OpenAI querying")
            continue
        elif query.lower() == 'langchain':
            mode = "langchain"
            print("Switched to LangChain QA")
            continue
        
        if mode == "langchain":
            query_engineering_knowledge(qa_chain, query)
        else:
            query_with_openai_directly(vectordb, query)


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Engineering PDF Knowledge Extraction with RAG")
    parser.add_argument("--pdf_dir", default="pdf_data", help="Directory containing PDF files")
    parser.add_argument("--load", action="store_true", help="Load existing vector database")
    parser.add_argument("--query", help="Query to run (if not provided, enter interactive mode)")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI directly instead of LangChain")
    
    args = parser.parse_args()
    
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Import dependencies
    if not import_dependencies():
        return
    
    # Set up API key
    setup_api_key()
    
    # Initialize embedding model
    embeddings = initialize_embedding_model()
    
    # Load or create vector database
    vectordb = None
    if args.load:
        vectordb = load_vector_db(embeddings)
        if not vectordb:
            print("Could not load vector database. Creating new one from PDFs...")
            vectordb = process_multiple_pdfs(args.pdf_dir, embeddings)
    else:
        vectordb = process_multiple_pdfs(args.pdf_dir, embeddings)
    
    if not vectordb:
        print("No vector database available. Please add PDFs to the pdf_data directory and try again.")
        return
    
    # Save project state
    save_project_state(vectordb)
    
    # Handle querying
    if args.query:
        if args.openai:
            query_with_openai_directly(vectordb, args.query)
        else:
            qa_chain = setup_rag_system(vectordb)
            query_engineering_knowledge(qa_chain, args.query)
    elif args.interactive or not args.query:
        interactive_mode(vectordb)


if __name__ == "__main__":
    main()