#!/usr/bin/env python3
"""
Seerah RAG Contextualizer - Islamic Knowledge Assistant
Uses Retrieval-Augmented Generation with Google Gemini embeddings
"""

import os
import time
import traceback
from dotenv import load_dotenv
from tqdm import tqdm

# Configuration
DATA_DIR = "seerah_data"  # Directory containing .txt/.pdf files
VECTOR_STORE_PATH = "vector_store"
OUTPUT_FILENAME = "seerah_rag_output.txt"

# Load environment
load_dotenv()

# Core imports
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Islamic-optimized prompt template
ISLAMIC_PROMPT = ChatPromptTemplate.from_template("""
Answer according to authentic Seerah sources only. If unsure, say "Allah knows best".

Question: {question} 
Context: {context}

Guidelines:
1. Base answers strictly on the context
2. Never speculate about the unseen (ghayb)
3. Cite sources when possible
4. If context is insufficient, respond: "I couldn't verify this from authentic Seerah sources"

Format:
[Verified Answer]: [Your answer]  
[Source Context]: [Relevant passage]
""")

# Known Islamic facts (fallback when documents lack info)
KNOWN_FACTS = {
    "revelation_period": Document(
        page_content="""
        The revelation period lasted approximately 23 years:
        - 13 years in Makkah
        - 10 years in Madinah
        First revelation came at age 40 in Cave Hira.
        """,
        metadata={"source": "known_facts"}
    ),
    "prophet_birth": Document(
        page_content="""
        Prophet Muhammad (PBUH) was born in the Year of the Elephant (570 CE) 
        in Makkah. His father Abdullah died before his birth, and mother Amina 
        died when he was six.
        """,
        metadata={"source": "known_facts"}
    )
}

def load_documents(data_directory):
    """Load documents from directory with error handling"""
    print(f"\nLoading documents from: {data_directory}")
    
    if not os.path.isdir(data_directory):
        print(f"Error: Directory '{data_directory}' not found")
        return []

    try:
        loader = DirectoryLoader(
            data_directory,
            glob="**/*[.txt,.pdf]",
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            loader_kwargs={"silent_errors": True}
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        return docs
        
    except Exception as e:
        print(f"\nDocument loading error: {e}")
        traceback.print_exc()
        return []

def split_text(documents):
    """Split documents into chunks with overlap"""
    if not documents:
        return []

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"\nText splitting error: {e}")
        traceback.print_exc()
        return []

def create_vector_store(chunks):
    """Create ChromaDB vector store with Gemini embeddings"""
    print("\nCreating vector store...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document",
        title="Seerah_of_Prophet_Muhammad"
    )
    
    # Initialize ChromaDB first
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    batch_size = 100 # Define a reasonable batch size

    for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks to vector store"):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch) # Add documents in batches
        # time.sleep(1) # Optional: add a small delay between batches if still facing issues

    print("✅ Vector store created")
    return vector_store

def get_retriever(vector_store):
    """Create a retriever with similarity threshold"""
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )

def enhanced_retriever(question, vector_store):
    """Retrieve docs with fallback to known facts"""
    retriever = get_retriever(vector_store)
    docs = retriever.invoke(question)
    
    # Add known facts if relevant
    if "revelation" in question.lower() and "year" in question.lower():
        docs.append(KNOWN_FACTS["revelation_period"])
    elif "birth" in question.lower() or "born" in question.lower():
        docs.append(KNOWN_FACTS["prophet_birth"])
        
    return docs


def format_docs(docs):
    """Format docs for context"""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" 
        for doc in docs
    )


# Define get_context outside initialize_qa_chain for clarity
def get_context(question: str, vector_store: Chroma) -> str:
    """Retrieves relevant documents and formats them for context."""
    docs = enhanced_retriever(question, vector_store)
    return format_docs(docs)


def initialize_qa_chain(vector_store: Chroma): # Pass vector_store as argument
    """Initialize the RAG QA chain"""
    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=1
    )
 

      # Define the chain using LCEL
    # 1. Pass the input question to both the context retriever and the prompt
    # 2. The context retriever uses the question and vector_store to get formatted context
    # 3. The prompt receives the question and the generated context
    # 4. The model generates the answer
    # 5. The output parser extracts the string response
    chain = (
        {"context": lambda q: get_context(q, vector_store), "question": RunnablePassthrough()}
        | ISLAMIC_PROMPT
        | model
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    # Pipeline
    documents = load_documents(DATA_DIR)
    chunks = split_text(documents)
    
    if chunks:
        # Initialize embeddings once
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document",
            title="Seerah_of_Prophet_Muhammad"
        )
        # Check if the vector store already exists
        if os.path.exists(VECTOR_STORE_PATH):
            print("Loading existing vector store...")
            vector_store = Chroma(persist_directory=VECTOR_STORE_PATH,embedding_function=embeddings)
        else:
            vector_store = create_vector_store(chunks)
        
        # Pass vector_store to the chain initializer
        qa_chain = initialize_qa_chain(vector_store)
               
        print("\n" + "="*50)
        print("Seerah RAG System Ready")
        print("Example questions:")
        print("- How old was the Prophet when he received revelation?")
        print("- Where was the first revelation revealed?")
        print("- How many years did the revelation take?")
        print("="*50 + "\n")
        
        while True:
            question = input("\nYour question about Seerah (or 'exit'): ")
            if question.lower() == 'exit':
                break
                
            response = qa_chain.invoke(question)
            print("\n" + response)