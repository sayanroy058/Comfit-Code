
import os
import logging
import PyPDF2
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentManager:
    """
    Manages document processing, indexing, and retrieval for the RAG system.
    """
    
    def __init__(self, documents_dir: str = "documents", vector_store_dir: str = "vector_store"):
        self.documents_dir = Path(documents_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.documents_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)
        
        # Initialize vector store components
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize the vector store for document indexing."""
        try:
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
            import os
            
            # Set OLLAMA_HOST environment variable if available
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "localhost:11434")
            os.environ['OLLAMA_HOST'] = ollama_base_url
            
            # Setup embedding model
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            # Setup LLM
            Settings.llm = Ollama(model="llama3:latest", request_timeout=600.0)
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Process a PDF file and extract text content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the processed text file, or None if processing failed
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            # Create output filename
            txt_filename = pdf_path.stem + '.txt'
            output_path = self.documents_dir / txt_filename
            
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Extract text from PDF
            text_content = self._extract_pdf_text(pdf_path)
            if not text_content:
                logger.error(f"No text content extracted from {pdf_path}")
                return None
            
            # Clean and save text
            cleaned_text = self._clean_text(text_content)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            logger.info(f"PDF processed successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_pages = []
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_pages.append(text)
                
                return "\n\n".join(text_pages)
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text."""
        # Remove page breaks and fix hyphenation
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'[.!?]\n', '. ', text)
        text = re.sub(r'[,;]\n', ', ', text)
        text = text.replace('\n', ' ')
        
        # Remove extra whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        
        # Remove page markers
        text = re.sub(r'--- PAGE \d+ ---', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def create_index(self, text_file_path: str, index_name: str = "default") -> bool:
        """
        Create a vector index from a text file.
        
        Args:
            text_file_path: Path to the text file
            index_name: Name for the index
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
            from llama_index.core.schema import TextNode
            
            text_path = Path(text_file_path)
            if not text_path.exists():
                logger.error(f"Text file not found: {text_path}")
                return False
            
            logger.info(f"Creating index '{index_name}' from {text_path.name}")
            
            # Load document
            reader = SimpleDirectoryReader(input_files=[str(text_path)])
            documents = reader.load_data()
            
            if not documents:
                logger.error("No documents loaded for indexing")
                return False
            
            # Add metadata
            for doc in documents:
                doc.metadata['filename'] = text_path.name
                doc.metadata['index_name'] = index_name
            
            # Create vector index
            index = VectorStoreIndex.from_documents(documents)
            
            # Save index
            index_path = self.vector_store_dir / f"{index_name}_index"
            index.storage_context.persist(persist_dir=str(index_path))
            
            logger.info(f"Index '{index_name}' created successfully: {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def load_index(self, index_name: str = "default"):
        """
        Load an existing vector index.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            VectorStoreIndex instance or None if loading failed
        """
        try:
            # First, check if there's a DuckDB file for this index
            duckdb_file = self.vector_store_dir / f"{index_name}.duckdb"
            if duckdb_file.exists():
                try:
                    from llama_index.vector_stores.duckdb import DuckDBVectorStore
                    from llama_index.core import VectorStoreIndex
                    
                    logger.info(f"Loading DuckDB vector store: {duckdb_file}")
                    vector_store = DuckDBVectorStore.from_local(str(duckdb_file))
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    logger.info(f"DuckDB vector store '{index_name}' loaded successfully")
                    return index
                except Exception as e:
                    logger.error(f"Error loading DuckDB vector store {index_name}: {e}")
            
            # Fallback: try to load as regular LlamaIndex storage context
            from llama_index.core import StorageContext, load_index_from_storage
            
            index_path = self.vector_store_dir / f"{index_name}_index"
            if not index_path.exists():
                logger.error(f"Index not found: {index_path}")
                return None
            
            # Load storage context
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            
            # Load index
            index = load_index_from_storage(storage_context)
            
            logger.info(f"Index '{index_name}' loaded successfully")
            return index
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None
    
    def query_index(self, query: str, index_name: str = "default", **kwargs) -> Optional[str]:
        """
        Query the vector index.
        
        Args:
            query: Query string
            index_name: Name of the index to query
            **kwargs: Additional query parameters
            
        Returns:
            Query response or None if query failed
        """
        try:
            index = self.load_index(index_name)
            if not index:
                return None
            
            # Create query engine
            query_engine = index.as_query_engine(**kwargs)
            
            # Execute query
            response = query_engine.query(query)
            
            logger.info(f"Query executed successfully: {query[:50]}...")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            return None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents."""
        documents = []
        
        for txt_file in self.documents_dir.glob("*.txt"):
            documents.append({
                "name": txt_file.name,
                "path": str(txt_file),
                "size": txt_file.stat().st_size,
                "modified": txt_file.stat().st_mtime
            })
        
        return documents
    
    def list_indexes(self) -> List[str]:
        """List all available indexes."""
        indexes = []
        
        for index_dir in self.vector_store_dir.glob("*_index"):
            if index_dir.is_dir():
                index_name = index_dir.name.replace("_index", "")
                indexes.append(index_name)
        
        return indexes
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document and its associated index."""
        try:
            # Delete text file
            txt_path = self.documents_dir / filename
            if txt_path.exists():
                txt_path.unlink()
            
            # Delete associated index
            index_name = txt_path.stem
            index_path = self.vector_store_dir / f"{index_name}_index"
            if index_path.exists():
                import shutil
                shutil.rmtree(index_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return False

# Global document manager instance
document_manager = DocumentManager(vector_store_dir="../vector_store")   


