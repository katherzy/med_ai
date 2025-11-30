import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "data/"
DB_FAISS_PATH = "/app/vector_db/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdf_files(data_path: str) -> List[Document]:
    """Load PDF files from the specified directory."""
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        loader = DirectoryLoader(
            data_path,
            glob='*.pdf',
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} PDF pages from {data_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        raise


def create_chunks(documents: List[Document], chunk_size: int = CHUNK_SIZE,
                  chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        raise


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Initialize and return the embedding model."""
    try:
        logger.info(f"Loading embedding model: {model_name}")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model loaded successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise


def create_vector_database(text_chunks: List[Document],
                           embedding_model: HuggingFaceEmbeddings,
                           db_path: str = DB_FAISS_PATH) -> None:
    """Create and save FAISS vector database."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        logger.info(f"Creating FAISS database at {db_path}")

        # Create vector database
        db = FAISS.from_documents(text_chunks, embedding_model)

        # Save to disk
        db.save_local(db_path)
        logger.info(f"Vector database saved successfully to {db_path}")
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise


def main():
    """Main execution function."""
    try:
        logger.info("Starting vector database creation process")

        # Load PDFs
        documents = load_pdf_files(DATA_PATH)

        if not documents:
            logger.warning("No documents found. Exiting.")
            return

        # Create chunks
        text_chunks = create_chunks(documents)

        # Get embedding model
        embedding_model = get_embedding_model()

        # Create and save vector database
        create_vector_database(text_chunks, embedding_model)

        logger.info("Vector database creation completed successfully")

    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        raise


if __name__ == "__main__":
    main()