#!/usr/bin/env python3
"""
Production-grade Seerah RAG Contextualizer
Islamic Knowledge Assistant using Retrieval-Augmented Generation
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Configuration and logging setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class Config:
    """Application configuration"""
    DATA_DIR = Path("seerah_data")  # Directory containing source files
    VECTOR_STORE_PATH = Path("vector_store")
    OUTPUT_FILENAME = "seerah_rag_output.txt"
    EMBEDDING_MODEL = "models/embedding-001"
    LLM_MODEL = "models/gemini-2.0-flash"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    RETRIEVER_K = 5
    RETRIEVER_SCORE_THRESHOLD = 0.5
    TEMPERATURE = 0.7

class DocumentProcessor:
    """Handles document loading and processing"""
    
    @staticmethod
    def load_documents(data_dir: Path) -> List[Document]:
        """Load documents from directory"""
        try:
            if not data_dir.is_dir():
                raise ValueError(f"Data directory not found: {data_dir}")
                
            loader = DirectoryLoader(
                str(data_dir),
                glob="**/*[.txt,.pdf]",
                loader_cls=UnstructuredFileLoader,
                show_progress=True,
                loader_kwargs={"silent_errors": True}
            )
            return loader.load()
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            raise

    @staticmethod
    def split_documents(documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        return splitter.split_documents(documents)

class VectorStoreManager:
    """Manages vector store operations"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            task_type="retrieval_document"
        )
        
    def create_store(self, chunks: List[Document]) -> Chroma:
        """Create new vector store"""
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(Config.VECTOR_STORE_PATH)
        )
        
    def load_store(self) -> Chroma:
        """Load existing vector store"""
        return Chroma(
            persist_directory=str(Config.VECTOR_STORE_PATH),
            embedding_function=self.embeddings
        )
        
    def store_exists(self) -> bool:
        """Check if vector store exists"""
        return Config.VECTOR_STORE_PATH.exists()

class KnowledgeRetriever:
    """Handles document retrieval with fallback to known facts"""
    
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
    
    def __init__(self, vector_store: Chroma):
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": Config.RETRIEVER_K,
                "score_threshold": Config.RETRIEVER_SCORE_THRESHOLD
            }
        )
        
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Retrieve documents with fallback to known facts"""
        docs = self.retriever.invoke(question)
        
        # Add known facts based on question content
        if "revelation" in question.lower() and "year" in question.lower():
            docs.append(self.KNOWN_FACTS["revelation_period"])
        elif "birth" in question.lower() or "born" in question.lower():
            docs.append(self.KNOWN_FACTS["prophet_birth"])
            
        return docs
        
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" 
            for doc in docs
        )

class QASystem:
    """Main QA system implementation"""
    
    PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
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
    
    def __init__(self, vector_store: Chroma):
        self.retriever = KnowledgeRetriever(vector_store)
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.chain = self._build_chain()
        
    def _build_chain(self):
        """Construct the RAG chain"""
        return (
            {
                "context": lambda q: self.retriever.format_docs(
                    self.retriever.get_relevant_documents(q)
                ),
                "question": RunnablePassthrough()
            }
            | self.PROMPT_TEMPLATE
            | self.llm
            | StrOutputParser()
        )
        
    def ask(self, question: str) -> str:
        """Get answer to a question"""
        try:
            return self.chain.invoke(question)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "An error occurred while processing your question."

def initialize_system() -> Optional[QASystem]:
    """Initialize the QA system"""
    try:
        # Document processing
        raw_docs = DocumentProcessor.load_documents(Config.DATA_DIR)
        chunks = DocumentProcessor.split_documents(raw_docs)
        
        if not chunks:
            logger.error("No documents available for processing")
            return None
            
        # Vector store management
        store_manager = VectorStoreManager()
        
        if store_manager.store_exists():
            logger.info("Loading existing vector store")
            vector_store = store_manager.load_store()
        else:
            logger.info("Creating new vector store")
            vector_store = store_manager.create_store(chunks)
            
        return QASystem(vector_store)
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return None

def main():
    """Main application entry point"""
    print("\nInitializing Seerah RAG System...")
    qa_system = initialize_system()
    
    if not qa_system:
        print("Failed to initialize system. Check logs for details.")
        return
        
    print("\n" + "="*50)
    print("Seerah RAG System Ready")
    print("Example questions:")
    print("- How old was the Prophet when he received revelation?")
    print("- Where was the first revelation revealed?")
    print("- How many years did the revelation take?")
    print("="*50 + "\n")
    
    while True:
        try:
            question = input("\nYour question about Seerah (or 'exit'): ").strip()
            if question.lower() == 'exit':
                break
                
            if not question:
                continue
                
            response = qa_system.ask(question)
            print("\n" + response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("\nAn error occurred. Please try again.")

if __name__ == "__main__":
    main()