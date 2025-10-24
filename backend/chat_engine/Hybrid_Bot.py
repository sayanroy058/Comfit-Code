
import os
import logging
import sys
import PyPDF2
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize
import hashlib
import traceback
import time
import asyncio
from functools import partial
from pathlib import Path
import shutil
import tempfile
import duckdb

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global Exception Hook for Tracebacks ---
def my_excepthook(type, value, traceback_obj):
    import traceback as tb
    tb.print_exception(type, value, traceback_obj)
    sys.exit(1)
sys.excepthook = my_excepthook

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from LlamaIndex and HTTP client libraries
logging.getLogger('llama_index').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Correcting the import for older LlamaIndex versions
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import MultiStepQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
# Corrected import path for Response for older LlamaIndex versions
from llama_index.core.base.response.schema import Response
from pydantic import BaseModel, Field

# --- Model Context Protocol Prompt Template ---
MODEL_CONTEXT_PROMPT = """
### User Query:
{query}

### Session Context:
{context_memory}

### Tool Outputs:
{tool_outputs}

### Scratchpad (Reasoning History):
{scratchpad}

### Instructions:
You are a helpful assistant. Use the context and available tool outputs to answer the user’s query as clearly and directly as possible.
Avoid mentioning any tools or internal steps. Provide only the final answer.

### Answer:
"""

# --- Model Context Protocol Prompt Template (With Thinking Process) ---
MODEL_CONTEXT_PROMPT_WITH_THINKING = """
### User Query:
{query}

### Session Context:
{context_memory}

### Tool Outputs:
{tool_outputs}

### Scratchpad (Reasoning History):
{scratchpad}

### Instructions:
You are Comfit Copilot, an AI expert in clothing comfort and fit.

IMPORTANT: When answering, you MUST structure your response in TWO sections:

**Section 1 - Thinking:**
- Start with "Thinking:" on a new line
- Provide your step-by-step reasoning process
- Explain which sources you're using and why
- Analyze the relevant information from the context
- Consider different aspects of the question
- Keep this section concise but informative (2-5 brief points)

**Section 2 - Final Answer:**
- Start with "Final Answer:" on a new line
- Provide a clear, direct, and complete answer to the user's question
- Base your answer on the reasoning from the Thinking section
- Avoid mentioning internal tools or steps

Example format:
Thinking:
1. The user is asking about 3D scanner technologies
2. From the local documents, I can see references to structured light and laser scanning
3. These are the two most common technologies mentioned in anthropometry contexts

Final Answer:
The two main technologies used in 3D scanners for anthropometry are structured-light scanning and laser scanning.

Now provide your response following this exact format.

### Answer:
"""

# --- LLM Setup ---
OPTIMAL_LLM_MODEL_NAME = "llama3"
OPTIMAL_LLM = Ollama(model=OPTIMAL_LLM_MODEL_NAME, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = OPTIMAL_LLM

# --- DuckDB Path Configuration ---
def load_duckdb_paths_from_env() -> Tuple[List[str], str]:
    """
    DEPRECATED: Loads DuckDB file paths from environment variables.
    This function is no longer used in the new vector store selection system.
    Kept for backward compatibility only.
    Returns a tuple of (list of all duckdb paths, extracted images duckdb path).
    """
    # Load the semicolon-separated list of DuckDB paths
    duckdb_paths_str = os.getenv("DUCKDB_PATHS", "")
    extracted_images_path = os.getenv("EXTRACTED_IMAGES_DUCKDB", "")
    
    if not duckdb_paths_str:
        logger.info("DUCKDB_PATHS not set in .env file. Using new vector store configuration system.")
        return [], ""
    
    # Split by semicolon and strip whitespace
    duckdb_paths = [path.strip() for path in duckdb_paths_str.split(';') if path.strip()]
    
    if not duckdb_paths:
        logger.info("No valid DuckDB paths found in DUCKDB_PATHS. Using new vector store configuration system.")
        return [], ""
    
    if not extracted_images_path:
        logger.info("EXTRACTED_IMAGES_DUCKDB not set in .env file. Using new vector store configuration system.")
    
    logger.info(f"Loaded {len(duckdb_paths)} DuckDB paths from environment variables (legacy)")
    return duckdb_paths, extracted_images_path

# Load DuckDB paths from environment (DEPRECATED - kept for backward compatibility)
DUCKDB_PATHS, EXTRACTED_IMAGES_DUCKDB_PATH = load_duckdb_paths_from_env()

# --- Helper to extract source filenames from LlamaIndex Response object ---
def _extract_source_filenames(response_obj: Response) -> List[str]:
    """Extracts unique filenames from the source nodes of a LlamaIndex Response object."""
    if not hasattr(response_obj, 'source_nodes') or not response_obj.source_nodes:
        logger.debug("Response object has no source_nodes attribute or source_nodes is empty")
        return []
    
    unique_filenames = set()
    for i, node in enumerate(response_obj.source_nodes):
        # Try multiple possible keys for filename
        filename = (
            node.metadata.get('filename') or 
            node.metadata.get('file_name') or 
            node.metadata.get('source') or
            node.metadata.get('file_path') or
            node.metadata.get('document_title') or
            node.metadata.get('title')
        )
        
        # Debug: log what metadata keys are available if no filename found
        if not filename and i == 0:  # Only log for first node to avoid spam
            logger.debug(f"Node metadata keys available: {list(node.metadata.keys())}")
        
        if filename:
            # Extract just the basename if it's a full path
            import os
            if '/' in str(filename) or '\\' in str(filename):
                filename = os.path.basename(str(filename))
            unique_filenames.add(str(filename))
    
    filenames_list = sorted(list(unique_filenames))
    logger.debug(f"Extracted {len(filenames_list)} unique source filenames")
    return filenames_list

# --- RAC (Retrieval-Augmented Correction) Implementation ---
class FactualClaimExtractor:
    """
    Extracts atomic factual claims from a given text using an LLM.
    These claims are then used for factual verification.
    """
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self, text: str) -> List[str]:
        logger.info("Extracting factual claims from the response...")
        # Prompt to instruct the LLM to extract atomic factual claims
        extraction_prompt = f"""
        Task: Extract atomic factual claims from the following text. 
        An atomic factual claim is a single, verifiable statement that can be true or false.
        
        Rules:
        1. Each claim should be independent and verifiable
        2. Break down complex sentences into simple facts
        3. Include numerical facts, dates, names, and specific details
        4. Ignore opinions, subjective statements, or procedural instructions
        5. Do NOT extract claims about the model's internal process or tools
        6. Return each claim on a separate line starting with "CLAIM:"
        
        Text to analyze:
        {text}
        
        Extract the factual claims:
        """
        try:
            response = self.llm.complete(extraction_prompt)
            claims = []
            for line in str(response).split('\n'):
                line = line.strip()
                if line.startswith('CLAIM:'):
                    claim = line.replace('CLAIM:', '').strip()
                    if claim and len(claim) > 10: # Filter out very short or empty claims
                        claims.append(claim)
            logger.info(f"Extracted {len(claims)} factual claims")
            return claims
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

class FactVerifier:
    """
    Verifies factual claims against local documents and web search results.
    It supports different retrieval methods and caches verification results.
    """
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.local_query_engine = local_query_engine
        self.Google_Search_tool = Google_Search_tool
        self.retrieval_method = retrieval_method # 'local', 'web', 'hybrid', 'automatic'
        self.verification_cache = {} # Cache to store previously verified claims
        self.reversal_min_confidence = 0.95 # Minimum confidence required for a factual reversal
        
    def _extract_search_terms(self, query: str, max_terms: int = 5) -> List[str]:
        """
        Extract key search terms from a query by removing stop words and limiting the number of keywords.
        
        Args:
            query: The user query string
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of extracted search terms
        """
        # Common stop words to filter out
        stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
            'where', 'how', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'of', 'for', 'with', 'about', 'me', 'my', 'show', 'display', 'give', 'image',
            'picture', 'diagram', 'figure', 'example'
        ])
        
        # Extract words from the query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words
        keywords = [word for word in words if word not in stop_words]
        
        # Limit to max_terms
        return keywords[:max_terms]
        
    def find_matching_image(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find an image that best matches the user's query by comparing query terms with image captions.
        
        Args:
            query: The user query asking for an image
            
        Returns:
            Dictionary containing image metadata if a match is found, None otherwise
        """
        # Extract search terms from the query
        search_terms = self._extract_search_terms(query)
        if not search_terms:
            logger.warning("No meaningful search terms extracted from query")
            return None
            
        logger.info(f"Searching for images with terms: {search_terms}")
        
        try:
            # Connect to the DuckDB database
            conn = duckdb.connect('extracted_images.duckdb')
            
            # Query the database for all images
            result = conn.execute("SELECT * FROM documents WHERE metadata_->>'type' = 'image'").fetchall()
            
            if not result:
                logger.warning("No images found in the database")
                return None
                
            # Process results to find matches
            matches = []
            for row in result:
                metadata = json.loads(row[2])  # Assuming metadata is in the third column
                
                if metadata.get('type') != 'image':
                    continue
                    
                caption = metadata.get('caption', '').lower()
                if not caption:
                    continue
                
                # Calculate match score based on keyword presence
                match_score = 0
                for term in search_terms:
                    if term.lower() in caption:
                        match_score += 1
                
                # Consider it a match if at least one term is found
                if match_score > 0:
                    matches.append({
                        'metadata': metadata,
                        'score': match_score / len(search_terms)  # Normalize score
                    })
            
            conn.close()
            
            # Sort matches by score in descending order
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            # If we have multiple good matches, use LLM to select the best one
            if len(matches) > 1 and matches[0]['score'] == matches[1]['score']:
                logger.info(f"Multiple matches found with equal scores, using LLM to select best match")
                
                # Prepare context for LLM
                context = f"Query: {query}\n\nAvailable images:\n"
                for i, match in enumerate(matches[:3]):  # Limit to top 3 matches
                    context += f"{i+1}. Caption: {match['metadata'].get('caption')}\n"
                
                # Ask LLM to select the best match
                prompt = f"""
                {context}
                
                Based on the user query and available image captions, which image (1, 2, or 3) is the most relevant match? 
                Respond with just the number of the best match.
                """
                
                response = self.llm.predict(prompt)
                try:
                    selected_index = int(response.strip()) - 1
                    if 0 <= selected_index < len(matches):
                        return matches[selected_index]['metadata']
                except:
                    # If LLM response parsing fails, fall back to highest score
                    pass
            
            # Return the highest scoring match if any found
            return matches[0]['metadata'] if matches else None
            
        except Exception as e:
            logger.error(f"Error finding matching image: {str(e)}")
            return None

    def _get_claim_hash(self, claim: str) -> str:
        """Generates a hash for a claim for caching purposes."""
        return hashlib.md5(claim.lower().encode('utf-8')).hexdigest()

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verifies a single factual claim by searching local and/or web sources.
        Returns a dictionary indicating support, confidence, evidence, and correction suggestions.
        """
        claim_hash = self._get_claim_hash(claim)
        if claim_hash in self.verification_cache:
            logger.info(f"Cache hit for claim: {claim[:50]}...")
            return self.verification_cache[claim_hash]
        
        logger.info(f"Verifying claim: {claim[:50]}... with retrieval method: {self.retrieval_method}")
        evidence = []
        web_sources = []  # To track structured web source links
        local_source_files = [] # To track specific local filenames

        use_local_source, use_web_source = False, False

        # Logic to determine which sources to use based on retrieval method
        if self.retrieval_method == "local":
            use_local_source = True
        elif self.retrieval_method == "web":
            use_web_source = True
        elif self.retrieval_method == "hybrid":
            use_local_source = True
            use_web_source = True
        elif self.retrieval_method == "automatic":
            # Heuristic to prioritize local search for specific keywords
            local_keywords = ["speed process", "anthropometry", "product fit", "sizing", "book", "document", "pdf", "chapter", "section"]
            if any(term in claim.lower() for term in local_keywords):
                use_local_source = True
            use_web_source = True # Always use web as a fallback for automatic mode

        if use_local_source:
            try:
                logger.info("Checking claim against local PDF content...")
                queries_to_try = [
                    claim,
                    f"What information is available about: {claim}",
                    f"Find details related to: {self._extract_search_terms(claim)}",
                    f"Does the document mention: {self._extract_search_terms(claim)}"
                ]
                local_evidence_found = False
                for query in queries_to_try:
                    try:
                        local_result = self.local_query_engine.query(query) # This returns a Response object
                        local_content = str(local_result).strip()
                        extracted_files = _extract_source_filenames(local_result)
                        
                        if (local_content and len(local_content) > 20 and
                                not any(phrase in local_content.lower() for phrase in [
                                        "i don't know", "no information", "not mentioned",
                                        "cannot find", "not available", "no details"])):
                            evidence.append({
                                'source': 'local_knowledge',
                                'content': local_content,
                                'confidence': 0.9, # Higher confidence for local, trusted data
                                'query_used': query,
                                'filenames': extracted_files # Store filenames here
                            })
                            local_source_files.extend(extracted_files)
                            local_evidence_found = True
                            logger.info(f"Local evidence found for query: {query[:50]}...")
                            break # Stop on first relevant local evidence
                    except Exception as e:
                        logger.warning(f"Local query failed for '{query[:30]}...': {e}")
                        continue
                if not local_evidence_found:
                    logger.info("No relevant local evidence found.")
            except Exception as e:
                logger.warning(f"Local verification failed: {e}")
        
        # Only use web if no local evidence found or if local is not chosen
        if use_web_source and not evidence: 
            try:
                logger.info(f"Searching web for: {claim[:50]}...")
                search_query = self._extract_search_terms(claim)
                # Call the Google Search tool and get both text and structured links
                web_result_text, web_links_from_search = self.Google_Search_tool.search(search_query) 
                if web_result_text and "No relevant search results" not in web_result_text:
                    evidence.append({
                        'source': 'web_search',
                        'content': web_result_text,
                        'confidence': 0.7, # Moderate confidence for web search
                        'query_used': search_query
                    })
                    web_sources.extend(web_links_from_search) # Store the structured web source links
                    logger.info(f"Web evidence found for query: {search_query}")
            except Exception as e:
                logger.warning(f"Web verification failed: {e}")
        
        evidence_sources = [e['source'] for e in evidence]
        logger.info(f"Evidence sources used: {evidence_sources}")
        verification_result = self._analyze_evidence(claim, evidence)
        
        # Store result in cache before returning
        result_to_cache = {
            'claim': claim,
            'is_supported': verification_result['is_supported'],
            'confidence': verification_result['confidence'],
            'evidence': evidence,
            'web_sources': web_sources, # Add web sources to result
            'local_source_files': sorted(list(set(local_source_files))), # Add unique local filenames
            'correction_suggestion': verification_result.get('correction', None),
            'warning': verification_result.get('warning', None)
        }
        self.verification_cache[claim_hash] = result_to_cache
        return result_to_cache
    
    def _extract_search_terms(self, claim: str) -> str:
        """Extracts key terms from a claim for more effective search queries."""
        words = claim.split()
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        return ' '.join(key_words[:5]) # Return up to 5 key words
        
    def find_matching_image(self, query: str) -> Dict[str, Any]:
        """
        Finds an image that matches the user's query by comparing with image captions.
        
        Args:
            query (str): User query asking for an image, e.g., "show me an image of wearable technologies"
            
        Returns:
            Dict with image information or None if no match found
        """
        logger.info(f"Finding image matching query: '{query}'")
        
        # Extract the actual search terms from queries like "show me an image of X" or "display the image of X"
        search_patterns = [
            r"show\s+(?:me\s+)?(?:an\s+)?(?:the\s+)?(?:example\s+)?(?:image|picture|diagram|figure)\s+(?:of|about|for|related\s+to)\s+(.*)",
            r"display\s+(?:the\s+)?(?:image|picture|diagram|figure)\s+(?:of|about|for|related\s+to)\s+(.*)",
            r"give\s+(?:me\s+)?(?:an\s+)?(?:example\s+)?(?:image|picture|diagram|figure)\s+(?:of|about|for|related\s+to)\s+(.*)"
        ]
        
        search_terms = query
        for pattern in search_patterns:
            match = re.search(pattern, query.lower())
            if match:
                search_terms = match.group(1)
                break
        
        logger.info(f"Extracted search terms: '{search_terms}'")
        
        # Connect to the DuckDB database
        try:
            db_path = EXTRACTED_IMAGES_DUCKDB_PATH
            if not db_path or not os.path.exists(db_path):
                logger.error(f"Extracted images DuckDB not found at: {db_path}")
                return "I apologize, but I cannot retrieve images at this time. The image database is not configured."
            con = duckdb.connect(db_path)
            
            # Fetch all rows from the 'documents' table
            rows = con.execute("SELECT node_id, text, metadata_ FROM documents").fetchall()
            
            # Process each row to find matching images
            matches = []
            for row in rows:
                node_id, text, metadata_json = row
                
                try:
                    metadata = json.loads(metadata_json)
                    
                    # Check if this is an image and has a caption
                    if metadata.get('type') == 'image' and 'caption' in metadata:
                        caption = metadata['caption']
                        
                        # Calculate a simple match score based on keyword overlap
                        search_keywords = set(search_terms.lower().split())
                        caption_keywords = set(caption.lower().split())
                        
                        # Remove common stop words
                        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
                        search_keywords = search_keywords - stop_words
                        caption_keywords = caption_keywords - stop_words
                        
                        # Calculate match score based on keyword overlap
                        common_words = search_keywords.intersection(caption_keywords)
                        if common_words:
                            match_score = len(common_words) / max(len(search_keywords), 1)
                            
                            # Add to matches if score is above threshold
                            if match_score > 0.2:  # Adjust threshold as needed
                                matches.append({
                                    'node_id': node_id,
                                    'caption': caption,
                                    'path': metadata.get('path', ''),
                                    'name': metadata.get('name', ''),
                                    'match_score': match_score
                                })
                except json.JSONDecodeError:
                    continue
            
            con.close()
            
            # If we have matches, select the best one
            if matches:
                # Sort by match score in descending order
                matches.sort(key=lambda x: x['match_score'], reverse=True)
                
                # Use LLM to select the most appropriate match if there are multiple good matches
                if len(matches) > 1 and matches[0]['match_score'] - matches[1]['match_score'] < 0.2:
                    top_matches = matches[:3]  # Consider top 3 matches
                    
                    # Create a prompt for the LLM to select the best match
                    options_text = "\n".join([f"{i+1}. Caption: {m['caption']}" for i, m in enumerate(top_matches)])
                    selection_prompt = f"""
                    Based on the user query: "{query}"
                    
                    Select the most relevant image from these options:
                    
                    {options_text}
                    
                    Analyze each option and select the number of the most relevant image that best matches the user's query.
                    Respond with ONLY the number (1, 2, or 3) of your selection.
                    """
                    
                    try:
                        response = OPTIMAL_LLM.complete(selection_prompt)
                        selection = str(response).strip()
                        
                        # Extract just the number from the response
                        selection_match = re.search(r'(\d+)', selection)
                        if selection_match:
                            selection_idx = int(selection_match.group(1)) - 1
                            if 0 <= selection_idx < len(top_matches):
                                best_match = top_matches[selection_idx]
                            else:
                                best_match = top_matches[0]  # Default to first match if invalid selection
                        else:
                            best_match = top_matches[0]  # Default to first match if no number found
                    except Exception as e:
                        logger.error(f"Error using LLM to select best image: {e}")
                        best_match = top_matches[0]  # Default to first match on error
                else:
                    best_match = matches[0]  # Use the highest scoring match
                
                logger.info(f"Found matching image: {best_match['name']} with score {best_match['match_score']}")
                return best_match
            else:
                logger.info("No matching images found")
                return None
                
        except Exception as e:
            logger.error(f"Error finding matching image: {e}")
            return None
    
    def retrieve_relevant_images(self, query: str, max_images: int = 2, selected_vector_store: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves up to max_images relevant images from the specified or default DuckDB
        based on semantic similarity with the user's query.
        
        Args:
            query (str): User query to match against image captions
            max_images (int): Maximum number of images to return (default: 2)
            selected_vector_store (str, optional): Name of the vector store to use
            
        Returns:
            List of dictionaries containing image information
        """
        logger.info(f"Retrieving relevant images for query: '{query}' from vector store: {selected_vector_store or 'default'}")
        
        # Import vector store config
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from vector_store_config import get_vector_store_path
        
        # Determine which DuckDB file to use - only from Vector_Store_Duckdb
        db_path = None
        
        if selected_vector_store:
            # Use the selected vector store from Vector_Store_Duckdb
            db_path = get_vector_store_path(selected_vector_store)
            if db_path and not os.path.exists(db_path):
                logger.warning(f"Selected vector store '{selected_vector_store}' not found at {db_path}")
                db_path = None
        
        if not db_path:
            logger.warning("No vector store selected or vector store not found. Please select a vector store from Vector_Store_Duckdb.")
            return []
        
        try:
            conn = duckdb.connect(db_path, read_only=True)
            
            # Check which schema exists
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            tables = [row[0] for row in conn.execute(tables_query).fetchall()]
            
            has_new_schema = 'pdf_content' in tables
            has_old_schema = 'documents' in tables
            
            if has_new_schema:
                # New schema with 'type' column
                # Detect actual column names (handle variations)
                try:
                    test_query = "SELECT * FROM pdf_content LIMIT 0"
                    test_result = conn.execute(test_query).description
                    column_names = [desc[0] for desc in test_result]
                    logger.debug(f"Detected columns in pdf_content: {column_names}")
                    
                    page_col = "pdf_page_no" if "pdf_page_no" in column_names else "page_no"
                    pdf_name_col = "original_pdf_name" if "original_pdf_name" in column_names else "pdf_name"
                except Exception as e:
                    logger.warning(f"Could not detect columns: {e}")
                    page_col = "page_no"
                    pdf_name_col = "pdf_name"
                
                rows = conn.execute(f"""
                    SELECT id, content, caption, {page_col} as page_no, {pdf_name_col} as pdf_name, type 
                    FROM pdf_content 
                    WHERE type = 'Image'
                """).fetchall()
                
                if not rows:
                    logger.info("No images found in database (new schema)")
                    conn.close()
                    return []
                
                # Process each row to find matching images (new schema)
                matches = []
                for row in rows:
                    row_id, content, caption, page_no, pdf_name, node_type = row
                    
                    if not caption:
                        continue
                    
                    # Calculate match score
                    query_keywords = set(query.lower().split())
                    caption_keywords = set(caption.lower().split())
                    
                    # Remove common stop words
                    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'which'}
                    query_keywords = query_keywords - stop_words
                    caption_keywords = caption_keywords - stop_words
                    
                    # Calculate match score based on keyword overlap
                    common_words = query_keywords.intersection(caption_keywords)
                    if common_words or len(query_keywords) == 0:  # Include all if query is too short
                        # Calculate score: ratio of matching keywords
                        if len(query_keywords) > 0:
                            match_score = len(common_words) / len(query_keywords)
                        else:
                            # If no query keywords, use caption relevance
                            match_score = 0.5
                        
                        # Boost score if there's a meaningful match
                        if match_score > 0:
                            # Clean the path - remove extra quotes if present
                            img_path = content or ''
                            if img_path.startswith('"') and img_path.endswith('"'):
                                img_path = img_path[1:-1]  # Remove surrounding quotes
                            
                            matches.append({
                                'node_id': row_id,
                                'caption': caption,
                                'path': img_path,
                                'name': caption,
                                'page': page_no or 'N/A',
                                'source_file': pdf_name or 'Unknown',
                                'match_score': match_score,
                            })
                
            elif has_old_schema:
                # Old schema with metadata_ column
                logger.debug(f"Using old schema for image retrieval")
                rows = conn.execute("SELECT node_id, text, metadata_ FROM documents").fetchall()
                
                if not rows:
                    logger.info("No images found in database (old schema)")
                    conn.close()
                    return []
                
                # Process each row to find matching images (old schema)
                matches = []
                for row in rows:
                    node_id, text, metadata_json = row
                    
                    try:
                        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                        
                        # Check if this is an image and has a caption
                        if metadata.get('type') == 'image' and 'caption' in metadata:
                            caption = metadata['caption']
                            
                            # Calculate match score
                            query_keywords = set(query.lower().split())
                            caption_keywords = set(caption.lower().split())
                            
                            # Remove common stop words
                            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'which'}
                            query_keywords = query_keywords - stop_words
                            caption_keywords = caption_keywords - stop_words
                            
                            # Calculate match score based on keyword overlap
                            common_words = query_keywords.intersection(caption_keywords)
                            if common_words or len(query_keywords) == 0:
                                # Calculate score: ratio of matching keywords
                                if len(query_keywords) > 0:
                                    match_score = len(common_words) / len(query_keywords)
                                else:
                                    match_score = 0.5
                                
                                # Boost score if there's a meaningful match
                                if match_score > 0:
                                    matches.append({
                                        'node_id': node_id,
                                        'caption': caption,
                                        'path': metadata.get('path', ''),
                                        'name': metadata.get('name', ''),
                                        'page': metadata.get('page', 'N/A'),
                                        'source_file': metadata.get('source_pdf', metadata.get('file_name', 'Unknown')),
                                        'match_score': match_score,
                                    })
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Error processing image metadata: {e}")
                        continue
            else:
                logger.error(f"No valid schema found in database: {db_path}")
                conn.close()
                return []
            
            if not matches:
                logger.info("No relevant images found for query")
                return []
            
            # Sort by match score in descending order
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top max_images results
            top_images = matches[:max_images]
            
            # Extract just the filename for display
            for img in top_images:
                if img['path']:
                    img['display_name'] = os.path.basename(img['path'])
                else:
                    img['display_name'] = img['name']
            
            logger.info(f"Found {len(top_images)} relevant image(s) for query")
            return top_images
            
        except Exception as e:
            logger.error(f"Error retrieving relevant images: {e}")
            return []
    
    def _analyze_evidence(self, claim: str, evidence: List[Dict]) -> Dict[str, Any]:
        """
        Analyzes the collected evidence to determine if the claim is supported,
        contradicted, or if there's insufficient evidence.
        Uses an LLM for nuanced analysis and correction suggestions.
        """
        if not evidence:
            return {
                'is_supported': False,
                'confidence': 0.0,
                'correction': None,
                'warning': "No evidence found to verify this claim."
            }
        
        # Separate local and web evidence for prioritized analysis
        local_evidence = [e for e in evidence if e['source'] == 'local_knowledge']
        web_evidence = [e for e in evidence if e['source'] == 'web_search']
        
        evidence_text = ""
        if local_evidence:
            evidence_text += "=== LOCAL DOCUMENT EVIDENCE ===\n"
            for e in local_evidence:
                filenames_str = f" (Files: {', '.join(e['filenames'])})" if e.get('filenames') else ""
                evidence_text += f"Source: {e['source']}{filenames_str} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        if web_evidence:
            evidence_text += "=== WEB SEARCH EVIDENCE ===\n"
            for e in web_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        
        # LLM prompt to analyze the claim against the evidence
        analysis_prompt = f"""
        Task: Analyze whether the CLAIM is supported by the EVIDENCE.
        CLAIM: {claim}
        EVIDENCE:
        {evidence_text}
        Instructions:
        1. Prioritize LOCAL DOCUMENT EVIDENCE for specific terms or definitions.
        2. Determine if SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.
        3. Provide confidence score (0.0 to 1.0).
        4. If contradicted, suggest a correction.
        5. For reversals (e.g., changing a positive statement to a negative one, or vice-versa),
           require confidence >= {self.reversal_min_confidence}.
        Respond:
        VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE]
        CONFIDENCE: [0.0-1.0]
        CORRECTION: [If contradicted, provide correction, else "None"]
        REASONING: [Brief explanation]
        """
        
        try:
            response = str(self.llm.complete(analysis_prompt))
            # Parse LLM response to extract verdict, confidence, and correction
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.5
            correction = None
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('VERDICT:'):
                    verdict = line.replace('VERDICT:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except ValueError:
                        confidence = 0.5 # Default if parsing fails
                elif line.startswith('CORRECTION:'):
                    correction_text = line.replace('CORRECTION:', '').strip()
                    if correction_text.lower() != 'none':
                        correction = correction_text
            
            # Boost confidence if local evidence was primary and relevant
            if local_evidence and verdict in ["SUPPORTED", "CONTRADICTED"]:
                confidence = min(confidence + 0.2, 1.0) # Add a small boost for local sources
            
            is_supported = verdict == "SUPPORTED"
            # Logic to detect potential factual reversals (positive to negative, or vice-versa)
            negative_markers = ["not", "no", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "never", "false", "untrue", "without"]
            original_has_neg = any(neg_m in claim.lower() for neg_m in negative_markers)
            correction_has_neg = any(neg_m in (correction or "").lower() for neg_m in negative_markers)
            
            # If a reversal is detected, check if confidence is high enough
            if correction and ((original_has_neg and not correction_has_neg) or (not original_has_neg and correction_has_neg)):
                if confidence < self.reversal_min_confidence:
                    logger.warning(f"Potential reversal detected but confidence ({confidence:.2f}) below threshold.")
                    correction = None # Suppress correction if confidence is too low
                    verdict = "INSUFFICIENT_EVIDENCE"
                    confidence = 0.5
            
            warning_message = None
            if not is_supported and confidence <= 0.6:
                # Flag responses with low confidence, even if no direct contradiction
                warning_message = f"Low confidence in claim: {claim}. Verdict: {verdict}. Confidence: {confidence:.2f}."
                logger.warning(warning_message)
            
            return {
                'is_supported': is_supported,
                'confidence': confidence,
                'correction': correction,
                'warning': warning_message
            }
        except Exception as e:
            logger.error(f"Error analyzing evidence: {e}")
            return {
                'is_supported': False,
                'confidence': 0.0,
                'correction': None,
                'warning': f"Error during evidence analysis: {e}"
            }

class RACCorrector:
    """
    Orchestrates the Retrieval-Augmented Correction (RAC) process.
    It extracts claims, verifies them, and applies corrections to the original response.
    """
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.claim_extractor = FactualClaimExtractor(llm)
        self.fact_verifier = FactVerifier(llm, local_query_engine, Google_Search_tool, retrieval_method)
        self.retrieval_method = retrieval_method
        self.correction_threshold = 0.5 # Minimum confidence to apply a correction
        self.uncertainty_threshold = 0.6 # Threshold to flag claims as uncertain
        self.local_priority = True # Not directly used in current logic, but can be for future weighting
        self.testing_mode = False # If true, corrections are not applied, only reported
        self.rac_enabled = True # Master switch for RAC

    def correct_response(self, original_response: str, apply_corrections: bool = True) -> Dict[str, Any]:
        """
        Applies RAC to an original LLM response.
        If RAC is disabled or no claims are extracted, returns the original response.
        """
        if not self.rac_enabled:
            logger.info("RAC is disabled, skipping correction.")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 1.0 # If RAC is off, assume full confidence
            }
        
        logger.info("Starting RAC correction process...")
        start_claim_extraction = time.perf_counter()
        claims = self.claim_extractor.extract_claims(original_response)
        end_claim_extraction = time.perf_counter()
        logger.info(f"Timing - Claim Extraction: {end_claim_extraction - start_claim_extraction:.4f} seconds")
        
        if not claims:
            logger.info("No factual claims extracted, returning original response")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 0.0 # Cannot assess confidence if no claims
            }
            
        verification_results = []
        corrections_needed = []
        uncertain_claims = []
        
        # Ensure the FactVerifier instance uses the current retrieval mode
        self.fact_verifier.retrieval_method = self.retrieval_method
        
        start_verification = time.perf_counter()
        for i, claim in enumerate(claims, 1):
            logger.info(f"Processing claim {i}/{len(claims)}: {claim[:50]}...")
            start_single_verify = time.perf_counter()
            result = self.fact_verifier.verify_claim(claim)
            end_single_verify = time.perf_counter()
            logger.info(f"Timing - Single Claim Verification ({i}): {end_single_verify - start_single_verify:.4f} seconds")
            
            verification_results.append(result)
            evidence_sources = [e['source'] for e in result['evidence']]
            logger.info(f"Claim {i} verification: {result['is_supported']}, confidence: {result['confidence']:.2f}, sources: {evidence_sources}")
            
            # If claim is not supported and confidence is above threshold, add to corrections
            if not result['is_supported'] and result['confidence'] > self.correction_threshold:
                if result['correction_suggestion']:
                    corrections_needed.append({
                        'original_claim': claim,
                        'correction': result['correction_suggestion'],
                        'confidence': result['confidence'],
                        'evidence_sources': evidence_sources,
                        'local_source_files': result.get('local_source_files', [])
                    })
                    logger.info(f"Correction needed for claim {i}")
                else:
                    logger.info(f"Claim {i} not supported but no correction available")
            
            # If confidence is below uncertainty threshold, flag it
            if result['confidence'] < self.uncertainty_threshold:
                uncertain_claims.append({
                    'claim': claim,
                    'confidence': result['confidence'],
                    'verdict': "SUPPORTED" if result['is_supported'] else "CONTRADICTED" if result['correction_suggestion'] else "INSUFFICIENT_EVIDENCE",
                    'warning': result.get('warning', 'Low confidence.')
                })
                logger.warning(f"Claim {i} is uncertain: {claim[:50]}... Confidence: {result['confidence']:.2f}")
        
        end_verification = time.perf_counter()
        logger.info(f"Timing - All Claims Verification: {end_verification - start_verification:.4f} seconds")
        
        corrected_response = original_response
        # Apply corrections only if not in testing mode and corrections are needed
        if apply_corrections and not self.testing_mode and corrections_needed:
            start_apply_corrections = time.perf_counter()
            corrected_response = self._apply_corrections(original_response, corrections_needed)
            end_apply_corrections = time.perf_counter()
            logger.info(f"Timing - Applying Corrections: {end_apply_corrections - start_apply_corrections:.4f} seconds")
            
        logger.info(f"RAC correction completed. Analyzed {len(claims)} claims, made {len(corrections_needed)} corrections")
        
        total_confidence = sum(res['confidence'] for res in verification_results)
        average_confidence = total_confidence / len(claims) if claims else 0.0 # Calculate average confidence
        
        return {
            'original_response': original_response,
            'corrected_response': corrected_response,
            'claims_analyzed': len(claims),
            'corrections_made': len(corrections_needed),
            'verification_results': verification_results,
            'corrections_applied': corrections_needed,
            'uncertain_claims': uncertain_claims,
            'average_confidence': average_confidence
        }
    
    def _apply_corrections(self, original_response: str, corrections: List[Dict]) -> str:
        """
        Uses an LLM to integrate the suggested corrections back into the original response,
        maintaining natural language flow.
        """
        correction_prompt = f"""
        Task: Apply corrections to the original response while maintaining its structure and flow.
        ORIGINAL RESPONSE:
        {original_response}
        CORRECTIONS TO APPLY:
        {chr(10).join([f"- Replace/correct: '{c['original_claim']}' -> '{c['correction']}'" for c in corrections])}
        Instructions:
        1. Integrate corrections naturally
        2. Maintain original tone and structure
        3. Ensure response flows well
        4. Don't add unnecessary information
        5. Keep response length similar
        6. Do NOT mention correction process
        Provide the corrected response:
        """
        try:
            corrected = str(self.llm.complete(correction_prompt))
            return corrected.strip()
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return original_response # Return original if correction application fails

# --- Google Search Tool ---
def validate_google_api_keys_from_env():
    """Validates presence of Google API keys from environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        logger.error("Google API keys not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file.")
        return None, None
    logger.info("Google API keys validated.")
    return google_api_key, google_cse_id

class GoogleCustomSearchTool:
    """
    A custom tool to perform Google Custom Search requests.
    It returns both formatted text results and structured links.
    """
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> tuple[str, list]:
        """
        Performs a Google Custom Search for the given query.
        Returns a formatted string of results and a list of structured link dictionaries.
        """
        logger.info(f"Google Search: '{query}'")
        start_web_api = time.perf_counter()
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": self.num_results
        }
        links = []  # To store structured link information
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            search_results = response.json()
            formatted_results = []
            if "items" in search_results:
                for i, item in enumerate(search_results["items"]):
                    title = item.get('title', 'N/A')
                    snippet = item.get('snippet', 'N/A')
                    link = item.get('link', 'N/A')
                    
                    # Store structured link info
                    links.append({
                        'title': title,
                        'url': link,
                        'snippet': snippet[:100] + "..." if len(snippet) > 100 else snippet
                    })
                    
                    # Format results for the LLM
                    formatted_results.append(
                        f"Result {i+1}: Title: {title}\n"
                        f"Snippet: {snippet}\n"
                        f"Link: {link}\n"
                        f"---"
                    )
                return "\n".join(formatted_results), links
            else:
                return "No relevant search results found.", []
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API error: {e}")
            return f"Error performing web search: {str(e)}", []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return "Error processing web search results.", []
        finally:
            end_web_api = time.perf_counter()
            logger.info(f"Timing - Google Web Search API Call: {end_web_api - start_web_api:.4f} seconds")

    def search_legacy(self, query: str) -> str:
        """
        Legacy method for backward compatibility, returning only the formatted text.
        This is used when the LlamaIndex agent expects a single string return from a tool.
        """
        result, _ = self.search(query)
        return result
    
    def search_images(self, query: str, num_images: int = 2) -> list:
        """
        Performs a Google Image Search for the given query.
        Returns a list of image dictionaries with title, link, and thumbnail.
        
        Args:
            query: Search query
            num_images: Number of images to return (default: 2)
        
        Returns:
            List of dictionaries containing image information
        """
        logger.info(f"Google Image Search: '{query}' (requesting {num_images} images)")
        start_image_search = time.perf_counter()
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "searchType": "image",  # Specify image search
            "num": min(num_images, 10),  # Google API max is 10 per request
            "safe": "active"  # Safe search
        }
        images = []
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            if "items" in search_results:
                for i, item in enumerate(search_results["items"]):
                    image_info = {
                        'title': item.get('title', 'N/A'),
                        'link': item.get('link', 'N/A'),  # Direct image URL
                        'thumbnail': item.get('image', {}).get('thumbnailLink', 'N/A'),
                        'context_link': item.get('image', {}).get('contextLink', 'N/A'),  # Source page
                        'width': item.get('image', {}).get('width', 0),
                        'height': item.get('image', {}).get('height', 0)
                    }
                    images.append(image_info)
                logger.info(f"Found {len(images)} images for query: '{query}'")
            else:
                logger.warning(f"No images found for query: '{query}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Image Search API error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in image search: {e}")
        finally:
            end_image_search = time.perf_counter()
            logger.info(f"Timing - Google Image Search API Call: {end_image_search - start_image_search:.4f} seconds")
        
        return images

# --- PDF Processing Functions ---
def clean_text(text):
    """
    Cleans extracted text from PDFs by removing hyphenation,
    standardizing newlines, removing extra spaces, and page markers.
    """
    # Remove hyphenation at line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace newlines after punctuation with a space
    text = re.sub(r'[.!?]\n', '. ', text)
    text = re.sub(r'[,;]\n', ', ', text)
    # Replace all remaining newlines with spaces
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # Remove PDF page markers if present
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    # Remove isolated page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def curate_pdf_to_text(pdf_path_str, output_dir):
    """
    Extracts text from a PDF, cleans it, and saves it to a text file.
    Uses a temporary file for processing to avoid issues with original paths.
    """
    pdf_path = Path(pdf_path_str)
    
    if not pdf_path.is_file():
        logger.critical(f"FATAL ERROR: PDF file not found at '{pdf_path_str}'. Exiting.")
        sys.exit(1)

    # Create a temporary directory for safe PDF copying and processing
    temp_dir = tempfile.mkdtemp()
    sanitized_filename = "temp_pdf.pdf"
    temp_pdf_path = Path(temp_dir) / sanitized_filename

    try:
        # Copy the original PDF to the temporary location
        shutil.copy(pdf_path, temp_pdf_path)
        logger.info(f"Copied original PDF to temporary path: {temp_pdf_path}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not copy PDF file from '{pdf_path_str}' to temporary location. Error: {e}. Exiting.")
        shutil.rmtree(temp_dir) # Clean up temp directory on error
        sys.exit(1)

    txt_filename = pdf_path.stem + '.txt'
    output_filepath = Path(output_dir) / txt_filename
    
    logger.info(f"Processing PDF: {pdf_path.name}...")
    full_text_pages = []
    try:
        with open(temp_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    full_text_pages.append(text)
            combined_text = "\n\n".join(full_text_pages)
            final_curated_text = clean_text(combined_text)
            if not final_curated_text.strip():
                logger.warning(f"Extracted text from '{pdf_path}' is empty. Skipping.")
                shutil.rmtree(temp_dir)
                return None
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                outfile.write(final_curated_text)
            logger.info(f"Curated and saved text to: {output_filepath}")
            return str(output_filepath)
    except PyPDF2.errors.PdfReadError:
        logger.critical(f"FATAL ERROR: Could not read PDF '{temp_pdf_path}'. Ensure it's a valid PDF. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL ERROR: Error processing '{temp_pdf_path}': {e}. Exiting.")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir) # Always clean up temporary directory


def load_nodes_from_duckdb(duckdb_path: str) -> List[TextNode]:
    """
    Loads TextNode objects from a DuckDB database.
    """
    if not os.path.exists(duckdb_path):
        logger.critical(f"FATAL ERROR: DuckDB file not found at '{duckdb_path}'. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loading nodes from DuckDB: {duckdb_path}")
    
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        
        # Query using the correct table and column names
        query = """
        SELECT 
            node_id,
            text,
            metadata_,
            embedding
        FROM documents
        """
        
        result = conn.execute(query).fetchall()
        conn.close()
        
        if not result:
            logger.warning(f"No nodes found in DuckDB file: {duckdb_path}")
            return []
        
        nodes = []
        for row in result:
            node_id, text, metadata_json, embedding = row
            
            # Parse metadata - it's already in JSON format
            metadata = metadata_json if isinstance(metadata_json, dict) else json.loads(metadata_json)
            
            # Create TextNode
            node = TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
                embedding=list(embedding) if embedding else None  # Convert array to list
            )
            nodes.append(node)
        
        logger.info(f"Loaded {len(nodes)} nodes from {os.path.basename(duckdb_path)}")
        return nodes
        
    except Exception as e:
        logger.error(f"ERROR: Could not load from DuckDB '{duckdb_path}': {e}")
        return []
    
def load_nodes_from_multiple_duckdb(duckdb_paths: List[str]) -> List[TextNode]:
    """
    Loads TextNode objects from multiple DuckDB databases.
    """
    all_nodes = []
    
    for duckdb_path in duckdb_paths:
        nodes = load_nodes_from_duckdb(duckdb_path)
        all_nodes.extend(nodes)
    
    if not all_nodes:
        logger.critical("FATAL ERROR: No nodes loaded from any DuckDB files. Exiting.")
        sys.exit(1)
    
    logger.info(f"Total nodes loaded from all DuckDB files: {len(all_nodes)}")
    return all_nodes


# OLD CODE - DISABLED: Now using vector_store_config.py for dynamic vector store selection
# This old initialization code is no longer used since we moved to the new system
# that allows users to select vector stores from Vector_Store_Duckdb via dropdown

# # Verify all DuckDB files exist (DUCKDB_PATHS loaded from environment variables above)
# for db_path in DUCKDB_PATHS:
#     if not os.path.exists(db_path):
#         logger.critical(f"FATAL ERROR: DuckDB file not found at '{db_path}'. Exiting.")
#         sys.exit(1)

# # Load nodes from all DuckDB files
# nodes = load_nodes_from_multiple_duckdb(DUCKDB_PATHS)
# logger.info("Creating VectorStoreIndex from DuckDB nodes...")
# try:
#     local_index = VectorStoreIndex(
#         nodes=nodes,
#         llm=OPTIMAL_LLM,
#         embed_model=Settings.embed_model,
#     )
#     # Configure the query engine to include source nodes
#     local_query_engine = local_index.as_query_engine(
#         llm=OPTIMAL_LLM,
#         response_mode="tree_summarize",
#         similarity_top_k=5,
#     )
#     logger.info("Local PDF data indexed successfully from DuckDB files.")
# except Exception as e:
#     logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex: {e}. Ensure Ollama models are running. Exiting.")
#     sys.exit(1)

def discover_duckdb_files(directory: str) -> List[str]:
    """
    Discovers all .duckdb files in a given directory.
    """
    if not os.path.exists(directory):
        logger.critical(f"FATAL ERROR: Directory not found: '{directory}'. Exiting.")
        sys.exit(1)
    
    duckdb_files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.duckdb')
    ]
    
    if not duckdb_files:
        logger.critical(f"FATAL ERROR: No .duckdb files found in '{directory}'. Exiting.")
        sys.exit(1)
    
    logger.info(f"Found {len(duckdb_files)} DuckDB files in {directory}")
    return duckdb_files


# Corrected filepaths list (ensure no invisible characters or missing commas)
# These paths are specific to the user's local system and should be adjusted if moved.
filepaths = [
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\_Multivariate Data Analysis_Hair.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DAnthropometryAndApplications.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DAnthropometryWearableProductDesign.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DLaserScanner.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\10.1201_9781003006091_previewpdf.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\9781439808801 (1).txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Materials & Manufacturing\biofunctional-textiles 8.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\Bodyspace Anthropometry, Ergonomics and the Design of the Work, Second Edition 1.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\DHM-HCII2019Book2.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\Douglas-C.-Montgomery-Design-and-Analysis-of-Experiments-Wiley-2012.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Ergonomics\Ergonomic_Office_Workstation_Design_that_Conforms_.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\GARIProceedings.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Standards & References\Human Dimension and Interior Space A Source Book of Design Reference Standards2.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\StatisticalModelHumanShapeAndPose.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Sustainability\865129978-Sustainable-Product-Design-And-Development-Anoop-Desai-Anil-Mital-pdf-download.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Fit & Sizing\9780429327803_googlepreview.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Fit & Sizing\ATOB-V12_Iss1_Article24.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Sustainability\A_Study_of_Sustainable_Product_Design_Evaluation_B.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\Design-of-head-mounteddisplays-Zhang.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\sensors-24-04616.txt",
    # r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\WEARABLE-TECHNOLOGIES3.txt",
]


def load_documents_for_indexing(files=None):
    """
    Loads and tags documents from a list of specified file paths into LlamaIndex Document objects.
    Performs a critical check for file existence.
    """
    if files is None:
        files = filepaths  # default to predefined list from global scope

    # Sanity check: ensure all specified files exist
    for f in files:
        if not os.path.exists(f):
            logger.critical(f"FATAL ERROR: Text file '{f}' not found. Exiting.")
            sys.exit(1)

    logger.info(f"Loading {len(files)} documents for indexing...")

    # Use SimpleDirectoryReader to load the text files
    reader = SimpleDirectoryReader(input_files=files, required_exts=[".txt"])
    documents = reader.load_data()

    if not documents:
        logger.critical("FATAL ERROR: No content loaded from provided files. Ensure files are not empty. Exiting.")
        sys.exit(1)

    # Attach metadata to each document for better retrieval and organization
    for doc in documents:
        doc_path = doc.metadata.get("file_path") or "unknown"
        doc.metadata['category'] = "BookContent" # General category for local documents
        doc.metadata['filename'] = os.path.basename(doc_path) # Original filename

    logger.info(f"Loaded {len(documents)} document segments in total.")
    return documents

# --- RAG Strategy Implementations ---
async def run_planning_workflow(query: str, agent_instance: ReActAgent, trace: List[str]) -> str:
    """
    Executes a query using the ReActAgent (planning workflow).
    The agent uses its tools to plan and execute steps to answer the query.
    """
    trace.append(f"Strategy: Planning Workflow - Agent thinking on '{query}'...")
    try:
        # Agent.chat() returns an AgentChatResponse which has a 'response' attribute
        response_obj = await asyncio.to_thread(agent_instance.chat, query)
        response = response_obj.response
        trace.append(f"Planning Workflow Raw Response: {response}")
        return response
    except Exception as e:
        trace.append(f"Error in Planning Workflow: {e}")
        logger.error(f"Error running planning workflow: {e}", exc_info=True)
        return "An error occurred during the planning workflow."

async def run_multi_step_query_engine_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Executes a query using a RouterQueryEngine, which selects between local and web query engines.
    Returns the response text, any web links collected, and local source filenames.
    """
    trace.append(f"Strategy: Multi-Step Query Engine - Routing '{query}'...")
    
    # Create QueryEngineTool for local data
    local_tool_choice = QueryEngineTool.from_defaults(
        query_engine=local_query_engine,
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "Use when the question relates to 'speed process', 'anthropometry', 'product fit', 'sizing', etc."
        ),
    )
    
    # Custom Google Query Engine that synthesizes results and returns links
    class GoogleQueryEngine:
        def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
            self.search_tool = search_tool_instance
            self.llm = llm
        
        async def aquery(self, query_str: str) -> Response:
            """Asynchronously queries Google Search and synthesizes an answer."""
            raw_search_result_text, links = await asyncio.to_thread(self.search_tool.search, query_str)
            
            if "No relevant search results" in raw_search_result_text:
                synthesized_answer = "No relevant information found on the web."
                return Response(response=synthesized_answer, metadata={"source": "Google Search", "links": links})

            synthesis_prompt = f"""
            Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
            Web Search Results:
            {raw_search_result_text}
            If the results do not contain a clear answer, state that.
            Provide only the answer, without referring to the search process or tools used.
            """
            try:
                synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                synthesized_answer = str(synthesized_answer)
            except Exception as e:
                logger.error(f"Error during Google search synthesis: {e}")
                synthesized_answer = "Could not synthesize an answer from web search results."
            # Return Response object including metadata for links
            return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis", "links": links})

        def query(self, query_str: str) -> Response:
            """Synchronous wrapper for aquery for compatibility."""
            return asyncio.run(self.aquery(query_str))

    google_qe_instance = GoogleQueryEngine(google_custom_search_instance, OPTIMAL_LLM)
    
    query_engine_tools = []
    # Add tools based on what's available in `tools_for_agent` (which reflects retrieval method selection)
    if any(tool.metadata.name == "local_book_qa" for tool in tools_for_agent):
        query_engine_tools.append(local_tool_choice)
    if any(tool.metadata.name == "google_web_search" for tool in tools_for_agent):
        query_engine_tools.append(
            QueryEngineTool.from_defaults(
                query_engine=google_qe_instance,
                description="Useful for general knowledge questions, current events, or anything requiring internet search."
            )
        )

    if not query_engine_tools:
        return "No relevant query engine available for the selected retrieval method.", [], []

    # Initialize RouterQueryEngine to select the best tool
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=query_engine_tools,
        llm=OPTIMAL_LLM
    )
    
    try:
        response_obj = await asyncio.to_thread(router_query_engine.query, query)
        response_text = str(response_obj)
        web_links_collected = response_obj.metadata.get("links", []) # Extract links from metadata for web search
        local_files_collected = _extract_source_filenames(response_obj) # Extract filenames for local search
        
        trace.append(f"Multi-Step Query Engine Raw Response: {response_text}")
        return response_text, web_links_collected, local_files_collected
    except Exception as e:
        trace.append(f"Error in Multi-Step Query Engine Workflow: {e}")
        logger.error(f"Error running multi_step_query_engine_workflow: {e}", exc_info=True)
        return "An error occurred during the multi-step query engine workflow.", [], []

async def run_multi_strategy_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Dict[str, Any]:
    """
    Executes both local RAG and web search queries, then synthesizes a combined answer.
    Returns the synthesized response and all collected web links.
    """
    trace.append(f"Strategy: Multi-Strategy Workflow - Executing multiple queries for '{query}'...")
    responses = []
    all_links_from_web_search = [] # To collect links specifically from web searches
    all_local_files = [] # To collect local filenames from local RAG
    
    # Determine which sources are enabled by the provided tools
    use_local_source = any(tool.metadata.name == "local_book_qa" for tool in tools_for_agent)
    use_web_source = any(tool.metadata.name == "google_web_search" for tool in tools_for_agent)
    
    if use_local_source:
        try:
            local_response_obj = await asyncio.to_thread(local_query_engine.query, query)
            local_response_text = str(local_response_obj)
            extracted_files = _extract_source_filenames(local_response_obj)
            
            responses.append(f"Local RAG result: {local_response_text}")
            all_local_files.extend(extracted_files)
            trace.append(f"Multi-Strategy: Local RAG executed. Response snippet: {local_response_text[:100]}...")
            if extracted_files:
                trace.append(f"Multi-Strategy: Local RAG sources: {', '.join(extracted_files)}")
        except Exception as e:
            responses.append(f"Local RAG error: {e}")
            trace.append(f"Multi-Strategy: Local RAG error: {e}")
            logger.warning(f"Error in Multi-Strategy local RAG: {e}")
    
    if use_web_source:
        # Re-using the GoogleQueryEngine logic for consistency in link extraction
        class GoogleQueryEngineForMultiStrategy:
            def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
                self.search_tool = search_tool_instance
                self.llm = llm
            async def aquery(self, query_str: str) -> Response:
                raw_search_result_text, links = await asyncio.to_thread(self.search_tool.search, query_str)
                if "No relevant search results" in raw_search_result_text:
                    synthesized_answer = "No relevant information found on the web."
                    return Response(response=synthesized_answer, metadata={"source": "Google Search", "links": links})
                synthesis_prompt = f"""
                Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
                Web Search Results:
                {raw_search_result_text}
                If the results do not contain a clear answer, state that.
                Provide only the answer, without referring to the search process or tools used.
                """
                try:
                    synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                    synthesized_answer = str(synthesized_answer)
                except Exception as e:
                    logger.error(f"Error during Google search synthesis (MultiStrategy): {e}")
                    synthesized_answer = "Could not synthesize an answer from web search results."
                return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis", "links": links})
            def query(self, query_str: str) -> Response:
                return asyncio.run(self.aquery(query_str))
        
        google_qe_instance = GoogleQueryEngineForMultiStrategy(google_custom_search_instance, OPTIMAL_LLM)
        try:
            web_response_obj = await asyncio.to_thread(google_qe_instance.query, query)
            web_response_text = str(web_response_obj)
            responses.append(f"Web Search result: {web_response_text}")
            trace.append(f"Multi-Strategy: Web Search executed. Response snippet: {web_response_text[:100]}...")
            all_links_from_web_search.extend(web_response_obj.metadata.get("links", [])) # Collect links
        except Exception as e:
            responses.append(f"Web Search error: {e}")
            trace.append(f"Multi-Strategy: Web Search error: {e}")
            logger.warning(f"Error in Multi-Strategy web search: {e}")
    
    combined_info = "\n\n".join(responses)
    if not combined_info.strip():
        return {"response": "No information found from any strategy.", "links": [], "local_files": []}
    
    # Synthesize a final answer from all collected information
    synthesis_prompt = f"""
    Based on the following information from various sources, provide a comprehensive answer to the question: "{query}".
    Information:
    {combined_info}
    Instructions:
    - Synthesize the information coherently.
    - If conflicting details exist, reconcile based on source authority.
    - Do not mention sources by name.
    - If no relevant information is available, state that.
    """
    try:
        final_answer = await asyncio.to_thread(OPTIMAL_LLM.complete, synthesis_prompt)
        final_answer = str(final_answer)
        trace.append(f"Multi-Strategy Synthesis Complete. Final Answer snippet: {final_answer[:100]}...")
        return {"response": final_answer, "links": all_links_from_web_search, "local_files": all_local_files}
    except Exception as e:
        trace.append(f"Error in Multi-Strategy Synthesis: {e}")
        logger.error(f"Error in multi-strategy synthesis: {e}", exc_info=True)
        return {"response": "An error occurred during multi-strategy synthesis.", "links": [], "local_files": []}

# --- Model Context Protocol Processing Function ---
async def process_model_context_query(
    query: str,
    context_memory: Dict[str, Any],
    tool_outputs: List[Dict],
    scratchpad: str,
    agent_instance: ReActAgent,
    rac_corrector_instance: 'RACCorrector',
    testing_mode: bool,
    suppress_threshold: float,
    flag_threshold: float,
    selected_rag_strategy: str,
    selected_retrieval_method: str,
    local_query_engine: Any,
    google_custom_search_instance: Any,
    tools_for_agent: List[FunctionTool],
    skip_rac: bool = False,
    selected_vector_store: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the entire response generation pipeline, including:
    - Preprocessing the query
    - Executing the selected RAG strategy (Planning, Multi-Step, Multi-Strategy, No Method)
    - Applying Retrieval-Augmented Correction (RAC)
    - Handling confidence-based suppression/flagging
    - Collecting and formatting source information
    """
    logger.info(f"Processing Model Context Query: '{query}' with strategy: {selected_rag_strategy}, retrieval: {selected_retrieval_method}")
    response_trace = [f"ModelContextQuery received: Query='{query}'"]
    response_trace.append(f"Selected RAG Strategy: {selected_rag_strategy}")
    response_trace.append(f"Selected Retrieval Method: {selected_retrieval_method}")
    response_trace.append(f"Selected Vector Store: {selected_vector_store or 'Default'}")

    # Check if this is an image request
    image_request_patterns = [
        r"show\s+(?:me\s+)?(?:an\s+)?(?:the\s+)?(?:example\s+)?(?:image|picture|diagram|figure)",
        r"display\s+(?:the\s+)?(?:image|picture|diagram|figure)",
        r"give\s+(?:me\s+)?(?:an\s+)?(?:example\s+)?(?:image|picture|diagram|figure)"
    ]
    
    is_image_request = any(re.search(pattern, query.lower()) for pattern in image_request_patterns)
    
    if is_image_request:
        response_trace.append("Detected image request, using image retrieval from selected vector store")
        # Create an instance of FactVerifier to use its retrieve_relevant_images method
        fact_verifier = FactVerifier(OPTIMAL_LLM, local_query_engine, google_custom_search_instance)
        matching_images = fact_verifier.retrieve_relevant_images(query, max_images=1, selected_vector_store=selected_vector_store)
        
        if matching_images:
            matching_image = matching_images[0]  # Get the top match
            image_path = matching_image.get('path', '')
            image_name = matching_image.get('name', '')
            image_caption = matching_image.get('caption', '')
            
            response_trace.append(f"Found matching image: {image_name}")
            
            # Extract just the filename from the full path for display
            # This ensures compatibility regardless of where the DuckDB was created
            import os
            display_filename = os.path.basename(image_path) if image_path else image_name
            
            # Create a response with the image information (using just filename for path)
            image_response = f"Here's the image you requested:\n\nImage: {display_filename}\nCaption: {image_caption}\nPath: {display_filename}"
            
            # Create a modified matching_image dict with just the filename for the path
            image_info_for_frontend = {
                'node_id': matching_image.get('node_id', ''),
                'caption': image_caption,
                'path': display_filename,  # Send only filename to frontend
                'name': display_filename,
                'match_score': matching_image.get('match_score', 0)
            }
            
            return {
                "final_answer": image_response,
                "trace": response_trace,
                "confidence_score": 0.9,
                "sources_used": {
                    "local_sources_count": 1,
                    "local_files": [display_filename],  # Use filename only
                    "web_sources_count": 0,
                    "web_links": [],
                    "used_local": True,
                    "used_web": False,
                    "image_info": image_info_for_frontend  # Send modified info with filename only
                }
            }
        else:
            response_trace.append("No matching image found")
            # Continue with normal processing if no image found
    
    # Initialize source tracking dictionaries
    sources_used = {
        'local_sources': [], # Tracks detailed info about local queries
        'web_sources': [],    # Tracks queries made to web search
        'web_links_used': [], # Stores actual URL details from web searches
        'local_files_used': [], # Stores unique filenames from local documents
        'web_images_used': [] # Stores image URLs from Google image search
    }

    start_total_process_mcp = time.perf_counter()
    try:
        # --- Preprocessing ---
        start_preprocess = time.perf_counter()
        processed_question = query # Currently, simple pass-through
        response_trace.append(f"Pre-processed query: '{processed_question}'")
        end_preprocess = time.perf_counter()
        response_trace.append(f"Timing - Preprocessing: {end_preprocess - start_preprocess:.4f} seconds")

        # --- RAG Strategy Execution ---
        start_rag_strategy = time.perf_counter()
        original_response_text = ""
        # These are reset per query as they represent the context for *this* specific interaction
        tool_outputs = []  
        scratchpad = ""
        
        # Enhanced tool wrappers to capture source usage
        # These wrappers intercept tool calls by the agent and log the source usage.
        class SourceTrackingLocalBookQA:
            def __init__(self, query_engine):
                self.query_engine = query_engine
            
            def __call__(self, query: str) -> str:
                logger.info(f"Local RAG: Querying for '{query}'")
                response_obj = self.query_engine.query(query) # Get Response object
                local_response_text = str(response_obj)
                extracted_files = _extract_source_filenames(response_obj)
                
                # Debug logging
                logger.info(f"Local RAG: Response has source_nodes: {hasattr(response_obj, 'source_nodes')}")
                if hasattr(response_obj, 'source_nodes'):
                    logger.info(f"Local RAG: Number of source nodes: {len(response_obj.source_nodes) if response_obj.source_nodes else 0}")
                logger.info(f"Local RAG: Extracted {len(extracted_files)} files: {extracted_files}")

                sources_used['local_sources'].append({
                    'query': query,
                    'source_type': 'PDF Documents',
                    'timestamp': time.time(),
                    'filenames': extracted_files
                })
                sources_used['local_files_used'].extend(extracted_files)
                
                # Append source filenames to the response text for the agent's context
                if extracted_files:
                    local_response_text += f"\n\nLocal Sources: {', '.join(extracted_files)}"
                return local_response_text

        class SourceTrackingWebSearch:
            def __init__(self, search_tool):
                self.search_tool = search_tool
            
            def __call__(self, query: str) -> str:
                logger.info(f"Web Search: Querying for '{query}'")
                result_text, links = self.search_tool.search(query) # Call the actual search tool
                
                sources_used['web_sources'].append({
                    'query': query,
                    'source_type': 'Web Search',
                    'timestamp': time.time()
                })
                sources_used['web_links_used'].extend(links) # Collect structured links
                
                # For web-only retrieval, also fetch images
                if selected_retrieval_method == "web":
                    logger.info(f"Web-only mode: Fetching images for query '{query}'")
                    images = self.search_tool.search_images(query, num_images=2)
                    sources_used['web_images_used'].extend(images)
                    logger.info(f"Added {len(images)} images to web_images_used")
                
                return result_text # Agent expects a string

        # Create dynamically wrapped tools based on which tools are active for this query
        # Filter tools based on selected_retrieval_method
        enhanced_tools = []
        for tool in tools_for_agent:
            if tool.metadata.name == "local_book_qa":
                # Only include local tool if retrieval method allows local search
                if selected_retrieval_method in ["local", "hybrid", "automatic"]:
                    enhanced_local_qa = SourceTrackingLocalBookQA(local_query_engine)
                    enhanced_tool = FunctionTool.from_defaults(
                        fn=enhanced_local_qa,
                        name="local_book_qa",
                        description=tool.metadata.description
                    )
                    enhanced_tools.append(enhanced_tool)
                    logger.info(f"Local tool included for retrieval method: {selected_retrieval_method}")
            elif tool.metadata.name == "google_web_search":
                # Only include web tool if retrieval method allows web search
                if selected_retrieval_method in ["web", "hybrid", "automatic"]:
                    enhanced_web_search = SourceTrackingWebSearch(google_custom_search_instance)
                    enhanced_tool = FunctionTool.from_defaults(
                        fn=enhanced_web_search,
                        name="google_web_search",
                        description=tool.metadata.description
                    )
                    enhanced_tools.append(enhanced_tool)
                    logger.info(f"Web search tool included for retrieval method: {selected_retrieval_method}")
        
        # Check if we have at least one tool available
        if not enhanced_tools:
            error_msg = f"No tools available for the selected retrieval method: {selected_retrieval_method}. Please select a valid retrieval method."
            logger.error(error_msg)
            return {
                "final_answer": error_msg,
                "trace": response_trace + [error_msg],
                "confidence_score": 0.0,
                "sources_used": {
                    "local_sources_count": 0,
                    "local_files": [],
                    "web_sources_count": 0,
                    "web_links": [],
                    "used_local": False,
                    "used_web": False
                }
            }
        
        logger.info(f"Initialized {len(enhanced_tools)} tool(s) for retrieval method: {selected_retrieval_method}")
        
        # Increase max_iterations for web-only mode since it might need more queries
        max_iterations = 50 if selected_retrieval_method == "web" else 30
        
        # Initialize an agent instance for *this specific query* with the appropriate tools
        # Use version-compatible approach for creating agent
        try:
            if hasattr(ReActAgent, 'from_tools'):
                # New API (llama-index >= 0.10.0)
                agent_instance_for_query = ReActAgent.from_tools(
                    llm=OPTIMAL_LLM,
                    tools=enhanced_tools,
                    verbose=False,
                    max_iterations=max_iterations
                )
            else:
                # Old API - use direct constructor
                agent_instance_for_query = ReActAgent(
                    tools=enhanced_tools,
                    llm=OPTIMAL_LLM,
                    verbose=False,
                    max_iterations=max_iterations
                )
            logger.info("ReAct Agent initialized for query processing")
        except Exception as e:
            logger.error(f"Failed to initialize ReActAgent for query: {e}")
            # Fallback: use the existing agent_instance if available
            agent_instance_for_query = agent_instance if agent_instance else None
            if not agent_instance_for_query:
                raise Exception(f"Cannot initialize agent for query: {e}")

        # Execute the selected RAG strategy
        if selected_rag_strategy == "planning_workflow" or selected_rag_strategy == "rac_enhanced_hybrid_rag":
            original_response_text = await run_planning_workflow(processed_question, agent_instance_for_query, response_trace)
            tool_outputs.append({"tool": "planning_workflow", "result": original_response_text})
        elif selected_rag_strategy == "multi_step_query_engine":
            # For multi_step, `run_multi_step_query_engine_workflow` returns text, web links, and local files.
            original_response_text, links_from_msqe, files_from_msqe = await run_multi_step_query_engine_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, enhanced_tools
            )
            tool_outputs.append({"tool": "multi_step_query_engine", "result": original_response_text})
            # Add links and local files collected by multi-step query engine
            sources_used['web_links_used'].extend(links_from_msqe)
            sources_used['local_files_used'].extend(files_from_msqe)
            if links_from_msqe:
                sources_used['web_sources'].append({
                    'query': processed_question,
                    'source_type': 'Web Search (Multi-Step QE)',
                    'timestamp': time.time()
                })
            if files_from_msqe:
                sources_used['local_sources'].append({
                    'query': processed_question,
                    'source_type': 'PDF Documents (Multi-Step QE)',
                    'timestamp': time.time(),
                    'filenames': files_from_msqe
                })
        elif selected_rag_strategy == "multi_strategy_workflow":
            # For multi_strategy, `run_multi_strategy_workflow` returns a dict with 'response', 'links', and 'local_files'.
            response_data = await run_multi_strategy_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, enhanced_tools
            )
            original_response_text = response_data['response']
            tool_outputs.append({"tool": "multi_strategy_workflow", "result": original_response_text})
            # Links and local files are explicitly returned by run_multi_strategy_workflow
            sources_used['web_links_used'].extend(response_data['links'])
            sources_used['local_files_used'].extend(response_data['local_files'])
            if response_data['links']:
                sources_used['web_sources'].append({
                    'query': processed_question,
                    'source_type': 'Web Search (Multi-Strategy)',
                    'timestamp': time.time()
                })
            if response_data['local_files']:
                 sources_used['local_sources'].append({
                    'query': processed_question,
                    'source_type': 'PDF Documents (Multi-Strategy)',
                    'timestamp': time.time(),
                    'filenames': response_data['local_files']
                })
        elif selected_rag_strategy == "no_method":
            # No specific RAG strategy - use direct query based on retrieval method
            
            if selected_retrieval_method == "web":
                # For web-only retrieval, use Google search directly
                logger.info("No method strategy with web retrieval - using Google search")
                result_text, links = google_custom_search_instance.search(processed_question)
                
                # Store web sources
                sources_used['web_sources'].append({
                    'query': processed_question,
                    'source_type': 'Web Search (Direct)',
                    'timestamp': time.time()
                })
                sources_used['web_links_used'].extend(links)
                
                # Fetch images for web-only mode
                try:
                    images = google_custom_search_instance.search_images(processed_question, num_images=2)
                    sources_used['web_images_used'].extend(images)
                    logger.info(f"Retrieved {len(images)} images from Google Image Search")
                    response_trace.append(f"Found {len(images)} relevant images via Google Image Search")
                except Exception as e:
                    logger.warning(f"Could not retrieve images from Google: {e}")
                
                # Synthesize answer from web results using LLM
                if "No relevant search results" in result_text:
                    original_response_text = "No relevant information found on the web."
                else:
                    synthesis_prompt = f"""Based on the following web search results, provide a comprehensive answer to the question: "{processed_question}".

Web Search Results:
{result_text}

Provide a detailed answer based only on the information from these web sources. Do not mention the search process."""
                    
                    try:
                        synthesized = await asyncio.to_thread(OPTIMAL_LLM.complete, synthesis_prompt)
                        original_response_text = str(synthesized)
                    except Exception as e:
                        logger.error(f"Error synthesizing web search results: {e}")
                        original_response_text = result_text
                
                tool_outputs.append({"tool": "google_web_search", "result": original_response_text})
                response_trace.append(f"Web search direct response with {len(links)} sources")
                
            elif local_query_engine:
                # For local/hybrid retrieval, use local query engine
                logger.info("No method strategy with local retrieval - using local query engine")
                response_obj = await asyncio.to_thread(local_query_engine.query, processed_question)
                original_response_text = str(response_obj)
                tool_outputs.append({"tool": "local_query_engine", "result": original_response_text})
                response_trace.append(f"Local query engine direct response: '{original_response_text[:200]}...'")
                
                # Extract source filenames if available
                extracted_files = []
                if hasattr(response_obj, 'source_nodes') and response_obj.source_nodes:
                    for node in response_obj.source_nodes:
                        if hasattr(node, 'metadata') and 'pdf_name' in node.metadata:
                            pdf_name = node.metadata['pdf_name']
                            if pdf_name not in extracted_files:
                                extracted_files.append(pdf_name)
                        elif hasattr(node, 'metadata') and 'original_pdf_name' in node.metadata:
                            pdf_name = node.metadata['original_pdf_name']
                            if pdf_name not in extracted_files:
                                extracted_files.append(pdf_name)
                
                # Track sources
                if extracted_files:
                    sources_used['local_sources'].append({
                        'query': processed_question,
                        'source_type': 'PDF Documents (Direct Query)',
                        'timestamp': time.time(),
                        'filenames': extracted_files
                    })
                    sources_used['local_files_used'].extend(extracted_files)
            else:
                original_response_text = "No documents are available for query."
                logger.warning("No query engine available for no_method strategy")
                response_trace.append(original_response_text)
        else:
            original_response_text = "Invalid RAG strategy selected."
            logger.error(original_response_text)

        end_rag_strategy = time.perf_counter()
        response_trace.append(f"Timing - RAG Strategy ({selected_rag_strategy}) Execution: {end_rag_strategy - start_rag_strategy:.4f} seconds")

        # --- RAC Application ---
        final_answer_content = original_response_text
        average_conf = 1.0 # Default confidence if RAC is disabled or no claims are found
        rac_web_sources = [] # To collect web sources specifically from RAC's verification step
        rac_local_files = [] # To collect local source files specifically from RAC's verification step

        # Skip RAC if explicitly requested (e.g., for title generation)
        if skip_rac:
            response_trace.append("RAC correction skipped as requested.")
        elif rac_corrector_instance.rac_enabled:
            start_rac = time.perf_counter()
            response_trace.append("Applying RAC (Retrieval-Augmented Correction)...")
            
            # Crucial: Set the retrieval method on the RAC corrector instance before running,
            # so RAC uses the same source preference as the main RAG strategy.
            rac_corrector_instance.retrieval_method = selected_retrieval_method
            
            rac_result = rac_corrector_instance.correct_response(original_response_text, apply_corrections=not testing_mode)
            
            # Collect web sources and local files that RAC used for verification
            for verification in rac_result.get('verification_results', []):
                if 'web_sources' in verification:
                    rac_web_sources.extend(verification['web_sources'])
                if 'local_source_files' in verification:
                    rac_local_files.extend(verification['local_source_files'])
            
            end_rac = time.perf_counter()
            response_trace.append(f"Timing - RAC Process: {end_rac - start_rac:.4f} seconds")
            response_trace.append(f"RAC Analysis: {rac_result['claims_analyzed']} claims checked.")
            
            if rac_result['corrections_made'] > 0:
                response_trace.append(f"  Corrections Applied: {rac_result['corrections_made']}")
                for corr in rac_result['corrections_applied']:
                    response_trace.append(f"    - Original: '{corr['original_claim'][:70]}...' -> Corrected: '{corr['correction'][:70]}...'")
            
            if rac_result['uncertain_claims']:
                response_trace.append(f"  Uncertain Claims Flagged: {len(rac_result['uncertain_claims'])}")
                for uc in rac_result['uncertain_claims']:
                    response_trace.append(f"    - Claim: '{uc['claim'][:70]}...' (Conf: {uc['confidence']:.2f})")
            
            average_conf = rac_result['average_confidence']
            final_answer_content = rac_result['corrected_response'] if rac_result['corrections_made'] > 0 else original_response_text
            tool_outputs.append({"tool": "rac_corrector", "result": final_answer_content})
            
            # --- Confidence Cascade (Suppression/Flagging) ---
            if average_conf < suppress_threshold:
                final_answer_content = f"❌ Response suppressed due to very low confidence ({average_conf:.2f})."
                response_trace.append(f"Confidence Cascade: Response Suppressed (Avg Confidence: {average_conf:.2f})")
            elif average_conf < flag_threshold:
                final_answer_content = f"⚠️ Low confidence in response ({average_conf:.2f}). Please use with caution.\n\n" + final_answer_content
                response_trace.append(f"Confidence Cascade: Response Flagged (Avg Confidence: {average_conf:.2f})")
            else:
                response_trace.append(f"Confidence Cascade: Response Accepted (Avg Confidence: {average_conf:.2f})")

        # Combine all unique web sources and local files collected from direct tool calls and RAC verification
        all_web_sources = sources_used['web_links_used'] + rac_web_sources
        unique_web_sources = []
        seen_urls = set()
        for source in all_web_sources:
            if isinstance(source, dict) and 'url' in source:
                if source['url'] and source['url'] != 'N/A' and source['url'] not in seen_urls:
                    unique_web_sources.append(source)
                    seen_urls.add(source['url'])
        
        all_local_filenames = sources_used['local_files_used'] + rac_local_files
        unique_local_filenames = sorted(list(set(all_local_filenames))) # Ensure unique and sorted
        
        # --- Retrieve relevant images based on retrieval method ---
        relevant_images = []
        
        if selected_retrieval_method == "web":
            # For web-only retrieval, use Google Image Search results
            relevant_images = sources_used['web_images_used']
            if relevant_images:
                response_trace.append(f"Retrieved {len(relevant_images)} image(s) from Google Image Search")
            else:
                response_trace.append("No images found via Google Image Search")
        elif selected_retrieval_method in ["local", "hybrid", "automatic"]:
            # For local/hybrid retrieval, use local database images
            try:
                fact_verifier_for_images = FactVerifier(OPTIMAL_LLM, local_query_engine, google_custom_search_instance, selected_retrieval_method)
                relevant_images = fact_verifier_for_images.retrieve_relevant_images(query, max_images=2, selected_vector_store=selected_vector_store)
                
                if relevant_images:
                    response_trace.append(f"Retrieved {len(relevant_images)} relevant image(s) from local database ({selected_vector_store or 'default'})")
                else:
                    response_trace.append("No relevant images found in local database")
            except Exception as e:
                logger.warning(f"Error retrieving images: {e}")
                response_trace.append(f"Warning: Could not retrieve images - {str(e)}")
                
        final_answer = final_answer_content
        
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration: {end_total_process_mcp - start_total_process_mcp:.4f} seconds")

        return {
            "final_answer": final_answer,
            "trace": response_trace,
            "confidence_score": average_conf,
            "sources_used": {
                "local_sources_count": len(unique_local_filenames), # Count unique local files
                "local_files": unique_local_filenames, # Pass unique filenames
                "web_sources_count": len(sources_used['web_sources']), # Still count distinct web queries
                "web_links": unique_web_sources,
                "used_local": len(unique_local_filenames) > 0,
                "used_web": len(unique_web_sources) > 0,
                "images": relevant_images  # Add retrieved images to the response
            }
        }
    except Exception as e:
        logger.error(f"Error in process_model_context_query: {e}", exc_info=True)
        
        # Even if there's an error, try to return any images that were fetched
        relevant_images = []
        if selected_retrieval_method == "web" and sources_used.get('web_images_used'):
            relevant_images = sources_used['web_images_used']
            logger.info(f"Recovered {len(relevant_images)} images despite error")
        
        # Prepare unique sources even in error case
        all_web_sources = sources_used.get('web_links_used', [])
        unique_web_sources = []
        seen_urls = set()
        for source in all_web_sources:
            if isinstance(source, dict) and 'url' in source:
                if source['url'] and source['url'] != 'N/A' and source['url'] not in seen_urls:
                    unique_web_sources.append(source)
                    seen_urls.add(source['url'])
        
        error_message = "I found some information and images for you from Google, but encountered an issue completing the full response."
        if relevant_images:
            error_message += f" However, I've included {len(relevant_images)} relevant images below."
        
        response_trace.append(f"ERROR: {e}")
        response_trace.append(traceback.format_exc())
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration (Error): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")
        
        return {
            "final_answer": error_message,
            "trace": response_trace,
            "confidence_score": 0.5,  # Some confidence since we have data
            "sources_used": {
                "local_sources_count": 0,
                "local_files": [],
                "web_sources_count": len(sources_used.get('web_sources', [])),
                "web_links": unique_web_sources,
                "used_local": False,
                "used_web": len(unique_web_sources) > 0,
                "images": relevant_images  # Include images even in error case
            }
        }

async def process_model_context_query_with_progress(
    query: str,
    context_memory: Dict[str, Any],
    tool_outputs: List[Dict],
    scratchpad: str,
    agent_instance: ReActAgent,
    rac_corrector_instance: 'RACCorrector',
    testing_mode: bool,
    suppress_threshold: float,
    flag_threshold: float,
    selected_rag_strategy: str,
    selected_retrieval_method: str,
    local_query_engine: Any,
    google_custom_search_instance: Any,
    tools_for_agent: List[FunctionTool],
    skip_rac: bool = False,
    selected_vector_store: Optional[str] = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Progress-aware version of process_model_context_query that sends detailed updates via callback.
    This version wraps the original function and intercepts log messages to send them as progress updates.
    """
    import logging
    import asyncio
    from collections import deque
    
    # Queue to hold log messages for async processing
    log_queue = deque()
    
    class ProgressLogHandler(logging.Handler):
        """Custom handler that queues log messages for async processing"""
        def emit(self, record):
            try:
                msg = self.format(record)
                log_queue.append(msg)
            except Exception:
                pass
    
    # Add custom handler to logger
    progress_handler = None
    if progress_callback:
        progress_handler = ProgressLogHandler()
        progress_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(progress_handler)
        
        # Send initial message
        await progress_callback("analyzing_query", f"Processing query: {query[:50]}...")
    
    async def send_queued_logs():
        """Background task to send queued log messages"""
        while True:
            try:
                if log_queue:
                    msg = log_queue.popleft()
                    if progress_callback:
                        await progress_callback("log", msg)
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error sending queued logs: {e}")
    
    # Start background task to send logs
    log_sender_task = None
    if progress_callback:
        log_sender_task = asyncio.create_task(send_queued_logs())
    
    try:
        # Call the original function
        result = await process_model_context_query(
            query=query,
            context_memory=context_memory,
            tool_outputs=tool_outputs,
            scratchpad=scratchpad,
            agent_instance=agent_instance,
            rac_corrector_instance=rac_corrector_instance,
            testing_mode=testing_mode,
            suppress_threshold=suppress_threshold,
            flag_threshold=flag_threshold,
            selected_rag_strategy=selected_rag_strategy,
            selected_retrieval_method=selected_retrieval_method,
            local_query_engine=local_query_engine,
            google_custom_search_instance=google_custom_search_instance,
            tools_for_agent=tools_for_agent,
            skip_rac=skip_rac,
            selected_vector_store=selected_vector_store
        )
        
        # Give time for remaining logs to be sent
        if progress_callback:
            await asyncio.sleep(0.1)
            # Send any remaining logs
            while log_queue:
                msg = log_queue.popleft()
                await progress_callback("log", msg)
            
            await progress_callback("finalizing", "Response complete!")
        
        return result
    finally:
        # Clean up
        if log_sender_task:
            log_sender_task.cancel()
            try:
                await log_sender_task
            except asyncio.CancelledError:
                pass
        
        # Remove the custom handler
        if progress_handler:
            logger.removeHandler(progress_handler)

def main():
    """
    Main function to set up and run the enhanced hybrid chatbot.
    Handles CLI interaction, RAG strategy selection, and RAC toggling.
    """
    # Check for dry-run mode
    testing_mode_enabled = "--dry-run" in sys.argv
    if testing_mode_enabled:
        logger.info("RAC Testing Mode (--dry-run) enabled.")
        sys.argv.remove("--dry-run")

    # Ensure curated data directory exists
    CURATED_DATA_SINGLE_BOOK_DIR = 'curated_data_single_book'
    os.makedirs(CURATED_DATA_SINGLE_BOOK_DIR, exist_ok=True)
    
    # # Load and index local PDF documents
    # documents = load_documents_for_indexing()
    # logger.info("Creating VectorStoreIndex for local PDF data...")
    # try:
    #     local_index = VectorStoreIndex.from_documents(
    #         documents,
    #         llm=OPTIMAL_LLM,
    #         embed_model=Settings.embed_model,
    #     )
    #     # Configure the query engine to include source nodes
    #     local_query_engine = local_index.as_query_engine(
    #         llm=OPTIMAL_LLM,
    #         # In older versions, 'response_mode' alone might not guarantee source nodes.
    #         # However, 'tree_summarize' is a good bet for our purpose.
    #         response_mode="tree_summarize",
    #         similarity_top_k=5,
    #     )
    #     logger.info("Local PDF data indexed successfully.")
    # except Exception as e:
    #     logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex: {e}. Ensure Ollama models are running. Exiting.")
    #     sys.exit(1)

    # Validate Google API keys
    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. Exiting.")
        sys.exit(1)
    
    # Initialize Google Custom Search Tool
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5 # Number of web search results to retrieve
    )

    # Initialize RAC Corrector
    rac_corrector = RACCorrector(
        llm=OPTIMAL_LLM,
        local_query_engine=local_query_engine,
        Google_Search_tool=Google_Search_instance
    )
    rac_corrector.testing_mode = testing_mode_enabled
    
    # Define Pydantic input schema for local RAG tool
    class LocalBookQAToolInput(BaseModel):
        query: str = Field(description="The question to ask about the PDF book content.")

    # Create LlamaIndex FunctionTool for local RAG
    def local_book_qa_function(query: str) -> str:
        """Function to expose local PDF querying to the LlamaIndex agent."""
        logger.info(f"Local RAG: Querying for '{query}'")
        start_local_rag_query = time.perf_counter()
        response_obj = local_query_engine.query(query) # Get the Response object
        end_local_rag_query = time.perf_counter()
        logger.info(f"Timing - Local RAG Query: {end_local_rag_query - start_local_rag_query:.4f} seconds")
        
        response_text = str(response_obj)
        extracted_files = _extract_source_filenames(response_obj)
        
        if extracted_files:
            return f"{response_text}\n\nLocal Sources: {', '.join(extracted_files)}"
        return response_text

    local_rag_tool = FunctionTool.from_defaults(
        fn=local_book_qa_function,
        name="local_book_qa",
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "The response will include the answer and the names of the local files it came from."
        ),
        fn_schema=LocalBookQAToolInput,
    )

    # Create LlamaIndex FunctionTool for Google Web Search
    Google_Search_tool_for_agent = FunctionTool.from_defaults(
        fn=Google_Search_instance.search_legacy,  
        name="google_web_search",
        description=(
            "Useful for general knowledge questions, current events, or anything requiring internet search."
        ),
    )

    # Combine all tools that the main agent can use
    tools_for_agent = [local_rag_tool, Google_Search_tool_for_agent]

    logger.info(f"Initializing ReAct Agent with LLM: {OPTIMAL_LLM_MODEL_NAME}...")
    # Initialize the main ReAct Agent that will orchestrate the responses
    agent = ReActAgent.from_tools(
        llm=OPTIMAL_LLM,
        tools=tools_for_agent,
        verbose=False, # Set to True for detailed agent internal steps (useful for debugging)
        max_iterations=30 # Limit agent's thinking iterations
    )
    
    logger.info("Initialized Enhanced ReAct Agent with RAC.")
    logger.info("\n--- Enhanced Hybrid Chatbot with Model Context Protocol READY ---")
    logger.info(f"Agent uses LLM: {OPTIMAL_LLM_MODEL_NAME}")
    logger.info(f"Tools available: {', '.join([t.metadata.name for t in tools_for_agent])}")
    logger.info(f"RAC enabled ({'Testing Mode' if testing_mode_enabled else 'Active'})")
    logger.info(f"Type your questions. Type 'exit' to quit.")
    logger.info(f"Commands: 'toggle_rac', 'rac_stats', 'set_mode [local|web|hybrid|automatic]'")
    
    # Initialize RAC to enabled by default
    rac_corrector.rac_enabled = True
    # Statistics for RAC performance tracking
    rac_stats = {
        'total_queries': 0,
        'corrected_queries': 0,
        'total_corrections': 0,
        'uncertain_claims_flagged': 0,
        'responses_suppressed': 0,
        'responses_flagged_low_confidence': 0
    }
    
    # Confidence thresholds for response handling
    SUPPRESS_THRESHOLD = 0.4 # Below this, response is suppressed
    FLAG_THRESHOLD = 0.6     # Below this, response is flagged for low confidence

    # Available RAG strategies
    RAG_STRATEGIES = {
        "1": "rac_enhanced_hybrid_rag", # Default, uses planning workflow + RAC
        "2": "planning_workflow",
        "3": "multi_step_query_engine",
        "4": "multi_strategy_workflow",
        "5": "no_method" # Pure agent chat, relying on its internal reasoning for tool use
    }
    STRATEGY_NAMES = {
        "rac_enhanced_hybrid_rag": "RAC Enhanced Hybrid RAG",
        "planning_workflow": "Planning Workflow",
        "multi_step_query_engine": "Multi-Step Query Engine",
        "multi_strategy_workflow": "Multi-Strategy Workflow",
        "no_method": "No Specific RAG Method"
    }
    current_rag_strategy = "1" # Default strategy on startup

    # Available retrieval methods for RAC and RAG strategies
    RETRIEVAL_METHODS = {
        "local": "Local Only (PDF)",
        "web": "Web Only (Google Search)",
        "hybrid": "Local and Web",
        "automatic": "Automatic" # Heuristic-based selection
    }
    current_retrieval_method = "hybrid" # Default retrieval method on startup
    
    context_memory = {} # Simple in-memory context for chat history (not persisted)
    
    # Auto-discover DuckDB files
    DUCKDB_DIRECTORY = r"D:\Sahithi\9_3_2025_ComFit\ComFit\vector_store"
    DUCKDB_PATHS = discover_duckdb_files(DUCKDB_DIRECTORY)

    # Load nodes from all DuckDB files
    nodes = load_nodes_from_multiple_duckdb(DUCKDB_PATHS)

    logger.info("Creating VectorStoreIndex from DuckDB nodes...")
    try:
        local_index = VectorStoreIndex(
            nodes=nodes,
            llm=OPTIMAL_LLM,
            embed_model=Settings.embed_model,
        )
        # Configure the query engine to include source nodes
        local_query_engine = local_index.as_query_engine(
            llm=OPTIMAL_LLM,
            response_mode="tree_summarize",
            similarity_top_k=5,
        )
        logger.info("Local PDF data indexed successfully from DuckDB files.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex: {e}. Ensure Ollama models are running. Exiting.")
        sys.exit(1)

    # Main conversational loop
    while True:
        print("\n" + "="*50)
        print("Select RAG Strategy:")
        for key, value in RAG_STRATEGIES.items():
            print(f"  {key}. {STRATEGY_NAMES[value]}")
        strategy_choice = input(f"Enter strategy number (currently using {STRATEGY_NAMES[RAG_STRATEGIES[current_rag_strategy]]}): ").strip().lower()
        
        # Handle commands or strategy selection
        if strategy_choice == 'exit':
            logger.info("Exiting Enhanced Hybrid Chatbot. Goodbye!")
            break
        elif strategy_choice == 'toggle_rac':
            rac_corrector.rac_enabled = not rac_corrector.rac_enabled
            print(f"RAC is now {'ENABLED' if rac_corrector.rac_enabled else 'DISABLED'}")
            continue
        elif strategy_choice == 'rac_stats':
            # Display RAC statistics
            print(f"\n--- RAC Statistics ---")
            print(f"Total queries processed: {rac_stats['total_queries']}")
            print(f"Queries with corrections: {rac_stats['corrected_queries']}")
            print(f"Total corrections made: {rac_stats['total_corrections']}")
            print(f"Total uncertain claims flagged: {rac_stats['uncertain_claims_flagged']}")
            print(f"Responses suppressed: {rac_stats['responses_suppressed']}")
            print(f"Responses flagged: {rac_stats['responses_flagged_low_confidence']}")
            correction_rate = rac_stats['corrected_queries'] / max(rac_stats['total_queries'], 1) * 100
            print(f"Correction rate: {correction_rate:.1f}%")
            continue
        elif strategy_choice.startswith('set_mode'):
            # Change retrieval method
            parts = strategy_choice.split()
            if len(parts) == 2 and parts[0] == 'set_mode':
                mode = parts[1]
                if mode in RETRIEVAL_METHODS:
                    current_retrieval_method = mode
                    print(f"Retrieval method set to: {mode.upper()}.")
                else:
                    print(f"Invalid mode. Use 'set_mode {list(RETRIEVAL_METHODS.keys())}'.")
            else:
                print("Invalid command format. Use 'set_mode [local|web|hybrid|automatic]'.")
            continue
        elif strategy_choice in RAG_STRATEGIES:
            current_rag_strategy = strategy_choice
            print(f"Selected strategy: {STRATEGY_NAMES[RAG_STRATEGIES[current_rag_strategy]]}")
        else:
            print("Invalid strategy selection.")
            continue

        print("\n" + "="*50)
        print("Select Retrieval Method:")
        for key, desc in RETRIEVAL_METHODS.items():
            print(f"  {key}: {desc}")
        retrieval_choice = input(f"Enter retrieval method (currently using '{current_retrieval_method}'): ").strip().lower()
        
        if retrieval_choice in RETRIEVAL_METHODS:
            current_retrieval_method = retrieval_choice
            print(f"Selected retrieval method: {RETRIEVAL_METHODS[current_retrieval_method]}")
        else:
            print("Invalid retrieval method selected. Keeping current method.")
        
        user_question = input("Enter your question: ").strip()
        print("\n" + "="*50)
        print("--- Agent's Response (via Model Context Protocol) ---")
        print("="*50)

        # Build tools list for the current query based on selected retrieval method
        tools_for_query = []
        if current_retrieval_method in ["local", "hybrid", "automatic"]:
            tools_for_query.append(local_rag_tool)
        if current_retrieval_method in ["web", "hybrid", "automatic"]:
            tools_for_query.append(Google_Search_tool_for_agent)
        
        if not tools_for_query:
            print("❌ No retrieval source selected. Please choose a method that includes a data source (e.g., 'local', 'web', 'hybrid', 'automatic').")
            continue

        try:
            start_time = time.time()
            # Call the main processing function
            mcp_response = asyncio.run(process_model_context_query(
                query=user_question,
                context_memory=context_memory, # Pass current chat context
                tool_outputs=[], # Placeholder, populated internally by MCP
                scratchpad="",    # Placeholder, populated internally by MCP
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=testing_mode_enabled,
                suppress_threshold=SUPPRESS_THRESHOLD,
                flag_threshold=FLAG_THRESHOLD,
                selected_rag_strategy=RAG_STRATEGIES[current_rag_strategy],
                selected_retrieval_method=current_retrieval_method,
                local_query_engine=local_query_engine,
                google_custom_search_instance=Google_Search_instance,
                tools_for_agent=tools_for_query # Pass filtered tools
            ))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
            
            # Update RAC statistics
            rac_stats['total_queries'] += 1
            if rac_corrector.rac_enabled:
                if "❌" in mcp_response["final_answer"]:
                    rac_stats['responses_suppressed'] += 1
                elif "⚠️" in mcp_response["final_answer"]:
                    rac_stats['responses_flagged_low_confidence'] += 1
            
            # Print the final answer
            print(mcp_response["final_answer"])
            
            # Display formatted sources information
            sources_info_str = format_sources_info(mcp_response["sources_used"])
            print(sources_info_str)

            # Print the detailed model context trace
            print("\n--- Model Context Trace ---")
            for step in mcp_response["trace"]:
                print(step)
            print("--- End Model Context Trace ---")

            # Update more RAC statistics based on trace content
            if rac_corrector.rac_enabled:
                if any("Corrections Applied:" in step for step in mcp_response["trace"]):
                    rac_stats['corrected_queries'] += 1
                if any("Uncertain Claims Flagged:" in step for step in mcp_response["trace"]):
                    rac_stats['uncertain_claims_flagged'] += 1

            # Store the current interaction in context memory
            context_memory[user_question] = mcp_response["final_answer"]
        except Exception as e:
            logger.error(f"Error during interaction loop: {e}", exc_info=True)
            print("An unhandled error occurred while processing your request. Please check logs for details.")
            
        print("="*50 + "\n")
    logger.info("Enhanced Hybrid Chatbot with Model Context Protocol Session Completed.")

def format_sources_info(sources_info: Dict[str, Any]) -> str:
    """
    Formats the sources information into a user-friendly string for display.
    Now includes specific local book names and images.
    """
    if not sources_info:
        return "\n📚 **Sources Used:** None"
    
    info_lines = ["\n📚 **Sources Used:**"]
    
    local_files = sources_info.get('local_files', [])
    if sources_info.get('used_local', False) and local_files:
        info_lines.append(f"  📄 **Local PDF Documents Referenced:**")
        for i, filename in enumerate(local_files[:5], 1): # Limit to top 5 for brevity
            info_lines.append(f"    {i}. {filename}")
        if len(local_files) > 5:
            info_lines.append(f"    ... and {len(local_files) - 5} more local files (not displayed)")
    elif sources_info.get('used_local', False) and not local_files:
        info_lines.append(f"  📄 Local PDF Documents: {sources_info['local_sources_count']} queries (no specific file names extracted)")

    if sources_info.get('used_web', False):
        info_lines.append(f"  🌐 Web Search: {sources_info['web_sources_count']} queries")
        
        web_links = sources_info.get('web_links', [])
        if web_links:
            info_lines.append(f"\n🔗 **Web Sources Referenced (Top 5):**")
            for i, link in enumerate(web_links[:5], 1):  # Limit to top 5 links for brevity
                title = link.get('title', 'Unknown Title')
                url = link.get('url', '')
                snippet = link.get('snippet', '')
                
                info_lines.append(f"  {i}. **{title}**")
                info_lines.append(f"       URL: {url}")
                if snippet:
                    info_lines.append(f"       Preview: {snippet}")
                info_lines.append("") # Add a blank line for readability
            
            if len(web_links) > 5:
                info_lines.append(f"  ... and {len(web_links) - 5} more web sources (not displayed)")
    
    # Display images
    images = sources_info.get('images', [])
    if images:
        info_lines.append(f"\n🖼️  **Images Retrieved ({len(images)}):**")
        for i, img in enumerate(images, 1):
            # Check if it's a Google Image (has 'link' key) or local image (has 'path' key)
            if isinstance(img, dict):
                if 'link' in img:
                    # Google Image
                    info_lines.append(f"  {i}. **{img.get('title', 'Image')}**")
                    info_lines.append(f"       Image URL: {img.get('link', 'N/A')}")
                    info_lines.append(f"       Thumbnail: {img.get('thumbnail', 'N/A')}")
                    if img.get('context_link') and img.get('context_link') != 'N/A':
                        info_lines.append(f"       Source Page: {img.get('context_link')}")
                elif 'path' in img or 'caption' in img:
                    # Local Image
                    info_lines.append(f"  {i}. **{img.get('name', img.get('path', 'Local Image'))}**")
                    if img.get('caption'):
                        info_lines.append(f"       Caption: {img.get('caption')}")
                    if img.get('path'):
                        info_lines.append(f"       File: {img.get('path')}")
                info_lines.append("") # Add blank line for readability
    
    if not sources_info.get('used_local', False) and not sources_info.get('used_web', False):
        info_lines.append("  ℹ️ No external sources were consulted.")
    
    return "\n".join(info_lines)

# Entry point of the script
if __name__ == "__main__":
    main()