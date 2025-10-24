import os
import sys
import time
import uuid
import asyncio
import logging
import duckdb
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


# Import core pieces from your Hybrid_Bot module
from .Hybrid_Bot import (
    process_model_context_query,
    RACCorrector,
    GoogleCustomSearchTool,
    VectorStoreIndex,
    Ollama,
    OllamaEmbedding,
    Settings,
    FunctionTool,
    validate_google_api_keys_from_env,
    _extract_source_filenames,
)

# Import ReActAgent separately to handle version differences
try:
    from llama_index.core.agent import ReActAgent
except ImportError:
    try:
        from llama_index.agent import ReActAgent
    except ImportError:
        # If still not found, we'll handle it in the initialization
        ReActAgent = None

from llama_index.core.schema import TextNode
from .document_manager import document_manager
# Import vector store configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vector_store_config import get_vector_store_path, get_all_vector_stores


# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (client.py) - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------
# Thinking Content Parser
# -----------------------
def parse_thinking_and_answer(response_text: str) -> tuple[Optional[str], str]:
    """
    Parses the response text to extract thinking process and final answer.
    
    Expected format:
        Thinking:
        <thinking content>
        
        Final Answer:
        <answer content>
    
    Returns:
        tuple: (thinking_content, final_answer)
        - thinking_content: str or None if no thinking section found
        - final_answer: str (the actual answer or original text if no sections found)
    """
    import re
    
    # Try to find Thinking and Final Answer sections
    thinking_pattern = r'Thinking:\s*\n(.*?)(?=Final Answer:|$)'
    answer_pattern = r'Final Answer:\s*\n(.*)'
    
    thinking_match = re.search(thinking_pattern, response_text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(answer_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    thinking_content = None
    final_answer = response_text  # Default to full text
    
    if thinking_match:
        thinking_content = thinking_match.group(1).strip()
        logger.info(f"Extracted thinking content: {len(thinking_content)} characters")
    
    if answer_match:
        final_answer = answer_match.group(1).strip()
        logger.info(f"Extracted final answer: {len(final_answer)} characters")
    elif thinking_match:
        # If we found thinking but no final answer, the rest is the answer
        final_answer = response_text[thinking_match.end():].strip()
    
    return thinking_content, final_answer


# -----------------------
# DuckDB Loading Functions
# -----------------------
def load_nodes_from_duckdb(duckdb_path: str) -> List[TextNode]:
    """Loads TextNode objects from a DuckDB database with unified schema (text + images)."""
    if not os.path.exists(duckdb_path):
        logger.error(f"DuckDB file not found: {duckdb_path}")
        return []
    
    logger.info(f"Loading nodes from DuckDB: {os.path.basename(duckdb_path)}")
    
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        
        # Check which schema exists by querying table information
        tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        tables = [row[0] for row in conn.execute(tables_query).fetchall()]
        
        has_new_schema = 'pdf_content' in tables
        has_old_schema = 'documents' in tables
        
        if has_new_schema:
            # Use the new unified schema
            # Detect actual column names (handle variations like pdf_page_no vs page_no)
            try:
                test_query = "SELECT * FROM pdf_content LIMIT 0"
                test_result = conn.execute(test_query).description
                column_names = [desc[0] for desc in test_result]
                logger.debug(f"Detected columns in pdf_content: {column_names}")
                
                # Map column name variations
                page_col = "pdf_page_no" if "pdf_page_no" in column_names else "page_no"
                pdf_name_col = "original_pdf_name" if "original_pdf_name" in column_names else "pdf_name"
            except Exception as e:
                logger.warning(f"Could not detect columns, using defaults: {e}")
                page_col = "page_no"
                pdf_name_col = "pdf_name"
            
            # Check if embedding column exists
            embedding_col = "embedding" if "embedding" in column_names else None
            
            if embedding_col:
                query = f"""
                SELECT 
                    id,
                    content,
                    caption,
                    {page_col} as page_no,
                    {pdf_name_col} as pdf_name,
                    type,
                    embedding
                FROM pdf_content
                WHERE type = 'Text'
                """
            else:
                # No embedding column - skip it
                query = f"""
                SELECT 
                    id,
                    content,
                    caption,
                    {page_col} as page_no,
                    {pdf_name_col} as pdf_name,
                    type,
                    NULL as embedding
                FROM pdf_content
                WHERE type = 'Text'
                """
                logger.warning(f"No 'embedding' column found in {duckdb_path}, using NULL for embeddings")
            
            result = conn.execute(query).fetchall()
            
            if not result:
                logger.warning(f"No text nodes found in: {duckdb_path}")
                conn.close()
                return []
            
            logger.info(f"✅ Loaded {len(result)} text nodes from {os.path.basename(duckdb_path)} (new schema)")
            
            nodes = []
            for row in result:
                row_id, content, caption, page_no, pdf_name, node_type, embedding = row
                
                # Create metadata
                metadata = {
                    "filename": pdf_name or "Unknown",
                    "page": page_no or "N/A",
                    "caption": caption or "",
                    "type": node_type
                }
                
                # Create TextNode
                node = TextNode(
                    id_=str(row_id),
                    text=content or "",
                    metadata=metadata,
                    embedding=list(embedding) if embedding else None
                )
                nodes.append(node)
            
            logger.info(f"Loaded {len(nodes)} text nodes from {os.path.basename(duckdb_path)} (new schema)")
            conn.close()
            return nodes
            
        elif has_old_schema:
            # Use the old schema
            logger.debug(f"Using old schema for {os.path.basename(duckdb_path)}")
            
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
                logger.warning(f"No nodes found in: {duckdb_path}")
                return []
            
            nodes = []
            for row in result:
                node_id, text, metadata_json, embedding = row
                metadata = metadata_json if isinstance(metadata_json, dict) else json.loads(metadata_json)
                
                node = TextNode(
                    id_=node_id,
                    text=text,
                    metadata=metadata,
                    embedding=list(embedding) if embedding else None
                )
                nodes.append(node)
            
            logger.info(f"Loaded {len(nodes)} nodes from {os.path.basename(duckdb_path)} (old schema)")
            return nodes
        else:
            # No recognized schema found
            logger.error(f"No recognized schema found in {duckdb_path}. Expected 'pdf_content' or 'documents' table.")
            conn.close()
            return []
        
    except Exception as e:
        logger.error(f"Error loading from DuckDB '{duckdb_path}': {e}")
        return []


def load_nodes_from_multiple_duckdb(duckdb_paths: List[str]) -> List[TextNode]:
    """Loads nodes from multiple DuckDB files."""
    all_nodes = []
    for path in duckdb_paths:
        nodes = load_nodes_from_duckdb(path)
        all_nodes.extend(nodes)
    
    logger.info(f"Total nodes loaded from all DuckDB files: {len(all_nodes)}")
    return all_nodes


def discover_duckdb_files(directory: str) -> List[str]:
    """Discovers all .duckdb files in a directory (non-recursive)."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    duckdb_files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.duckdb')
    ]
    
    logger.info(f"Found {len(duckdb_files)} DuckDB files in {directory}")
    return duckdb_files


def discover_duckdb_files_recursive(directory: str) -> List[str]:
    """Recursively discovers all .duckdb files in a directory and subdirectories."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    duckdb_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.duckdb'):
                full_path = os.path.join(root, f)
                duckdb_files.append(full_path)
                logger.info(f"Found DuckDB file: {full_path}")
    
    logger.info(f"Found total {len(duckdb_files)} DuckDB files in {directory} and subdirectories")
    return duckdb_files


# -----------------------
# Normalization helpers
# -----------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


RAG_STRATEGY_MAP = {
    # canonical
    "rac_enhanced_hybrid_rag": "rac_enhanced_hybrid_rag",
    "planning_workflow": "planning_workflow",
    "multi_step_query_engine": "multi_step_query_engine",
    "multi_strategy_workflow": "multi_strategy_workflow",
    "no_method": "no_method",
    # common labels / UI strings
    "no specific rag method": "no_method",
    "no specific method": "no_method",
    "none": "no_method",
    "default": "rac_enhanced_hybrid_rag",
}


RETRIEVAL_METHOD_MAP = {
    # canonical
    "local": "local",
    "web": "web",
    "hybrid": "hybrid",
    "automatic": "automatic",
    # common labels / UI strings
    "local context only": "local",
    "local only (pdf)": "local",
    "local only": "local",
    "web only": "web",
    "internet": "web",
    "web search only": "web",
    "web searched context only": "web",  # Frontend sends this
    "google search": "web",
    "google": "web",
    "hybrid context": "hybrid",
    "smart retrieval": "automatic",
}


DEFAULT_RAG_STRATEGY = "multi_step_query_engine"
DEFAULT_RETRIEVAL = "hybrid"


# -----------------------
# Chat Engine
# -----------------------
class ChatEngine:
    def __init__(self):
        """
        Initializes all necessary components from Hybrid_Bot.py.
        Runs once on app startup.
        """
        logger.info("Initializing core components from hybrid_bot.py...")

        load_dotenv()

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = "llama3"
        self.local_query_engine = None
        self.google_search_instance = None
        self.rac_corrector = None
        self.agent = None
        self.tools_for_agent: List[FunctionTool] = []

        # Determine testing mode from command line arguments
        self.testing_mode_enabled = "--dry-run" in sys.argv
        if self.testing_mode_enabled:
            sys.argv.remove("--dry-run")

        # LlamaIndex global settings
        Settings.llm = Ollama(model=self.default_model, request_timeout=600.0, base_url=self.ollama_base_url)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

        try:
            # 1) Load & index local documents from DuckDB - using Vector_Store_Duckdb only
            # Use Vector_Store_Duckdb folder - recursively find all .duckdb files
            vector_store_base = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "..",
                "Vector_Store_Duckdb"
            ))
            logger.info(f"Looking for DuckDB files in: {vector_store_base}")
            duckdb_paths = discover_duckdb_files_recursive(vector_store_base)
            
            if not duckdb_paths:
                logger.warning("No DuckDB files found in Vector_Store_Duckdb. Local query engine will not be available.")
                self.local_query_engine = None
            else:
                nodes = load_nodes_from_multiple_duckdb(duckdb_paths)
                
                if nodes:
                    logger.info("Creating VectorStoreIndex from DuckDB nodes...")
                    self.local_index = VectorStoreIndex(
                        nodes=nodes,
                        llm=Settings.llm,
                        embed_model=Settings.embed_model
                    )
                    self.local_query_engine = self.local_index.as_query_engine(
                        llm=Settings.llm,
                        response_mode="tree_summarize",
                        similarity_top_k=5,
                    )
                    logger.info("Local documents indexed from DuckDB and query engine created.")
                else:
                    logger.warning("No nodes loaded from DuckDB files.")
                    self.local_query_engine = None

            # 2) Google Search Tool
            google_api_key, google_cse_id = validate_google_api_keys_from_env()
            if not (google_api_key and google_cse_id):
                raise ValueError("Google API keys not configured.")
            self.google_search_instance = GoogleCustomSearchTool(
                api_key=google_api_key,
                cse_id=google_cse_id,
                num_results=5
            )
            logger.info("Google Search tool initialized.")

            # 3) RAC Corrector
            self.rac_corrector = RACCorrector(
                llm=Settings.llm,
                local_query_engine=self.local_query_engine,
                Google_Search_tool=self.google_search_instance
            )
            self.rac_corrector.testing_mode = self.testing_mode_enabled
            logger.info("RAC Corrector initialized.")

            # 4) Tools for agent
            def _local_book_qa_function(q: str) -> str:
                if self.local_query_engine is None:
                    return "Local search is not available (no documents loaded)."
                try:
                    resp_obj = self.local_query_engine.query(q)
                    text = str(resp_obj)
                    files = [n.metadata.get('filename') for n in getattr(resp_obj, 'source_nodes', []) if n.metadata.get('filename')]
                    if files:
                        text += "\n\nLocal Sources: " + ", ".join(sorted(set(files)))
                    return text
                except Exception as e:
                    logger.warning(f"Local RAG tool error: {e}")
                    return "Local search failed for this query."

            local_rag_tool = FunctionTool.from_defaults(
                fn=_local_book_qa_function,
                name="local_book_qa",
                description="Useful for questions specifically about the content of the provided PDF documents."
            )

            google_search_tool_for_agent = FunctionTool.from_defaults(
                fn=self.google_search_instance.search_legacy,
                name="google_web_search",
                description="Useful for general knowledge questions, current events, or anything requiring internet search."
            )

            self.tools_for_agent = [local_rag_tool, google_search_tool_for_agent]

            # 5) Main ReAct Agent
            # Note: The agent initialization is now handled dynamically in generate_response
            # to avoid startup failures. The agent will be created when needed.
            self.agent = None
            logger.info("Agent will be initialized on first use (lazy loading).")
        except Exception as e:
            logger.critical(f"FATAL ERROR during ChatEngine initialization: {e}", exc_info=True)
            sys.exit(1)

    def _ensure_agent_initialized(self):
        """
        Lazily initialize the ReAct agent if not already initialized.
        This avoids startup failures and handles version differences.
        """
        if self.agent is not None:
            return
        
        logger.info("Initializing ReAct Agent...")
        try:
            # Try using from_tools (newer API)
            if hasattr(ReActAgent, 'from_tools'):
                self.agent = ReActAgent.from_tools(
                    llm=Settings.llm,
                    tools=self.tools_for_agent,
                    verbose=False,
                    max_iterations=30
                )
                logger.info("ReAct Agent initialized using from_tools()")
            else:
                # Try direct initialization
                self.agent = ReActAgent(
                    llm=Settings.llm,
                    tools=self.tools_for_agent,
                    verbose=False,
                    max_iterations=30
                )
                logger.info("ReAct Agent initialized using direct constructor")
        except Exception as e:
            logger.error(f"Failed to initialize ReAct Agent: {e}")
            logger.warning("Agent functionality will be limited. Continuing without agent...")
            # Set a dummy agent that will be checked before use
            self.agent = None

    def create_query_engine_for_vector_store(self, vector_store_name: str):
        """
        Creates a query engine for a specific vector store file.
        
        Args:
            vector_store_name: Display name of the vector store (e.g., "PhD Thesis LovatoC")
        
        Returns:
            Query engine instance or None if vector store not found
        """
        if not vector_store_name:
            logger.warning("No vector store name provided, using default query engine")
            return self.local_query_engine
        
        # Get the path to the selected vector store
        vector_store_path = get_vector_store_path(vector_store_name)
        
        if not vector_store_path:
            logger.error(f"Vector store '{vector_store_name}' not found in configuration")
            return self.local_query_engine
        
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store file does not exist: {vector_store_path}")
            return self.local_query_engine
        
        try:
            logger.info(f"Creating query engine for vector store: {vector_store_name}")
            nodes = load_nodes_from_duckdb(vector_store_path)
            
            if not nodes:
                logger.warning(f"No nodes loaded from {vector_store_name}")
                return self.local_query_engine
            
            # Create index from nodes
            index = VectorStoreIndex(
                nodes=nodes,
                llm=Settings.llm,
                embed_model=Settings.embed_model
            )
            
            # Create query engine
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                response_mode="tree_summarize",
                similarity_top_k=5,
            )
            
            logger.info(f"Successfully created query engine for {vector_store_name} with {len(nodes)} nodes")
            return query_engine
            
        except Exception as e:
            logger.error(f"Error creating query engine for {vector_store_name}: {e}")
            return self.local_query_engine

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        preset: str,
        temperature: float,
        user_id: str,
        rag_method: str,
        retrieval_method: str,
        selected_vector_store: Optional[str] = None,
        skip_rac: bool = False,
    ) -> tuple[str, int, Optional[Dict[str, Any]], Optional[str]]:
        """
        Orchestrates the RAG pipeline and returns (response_text, duration_ms, image_info, thinking_content).
        """
        logger.info(f"Received request for conversation_id: {conversation_id}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"RAW retrieval_method received: '{retrieval_method}'")
        logger.info(f"RAW rag_method received: '{rag_method}'")
        logger.info(f"Selected vector store: '{selected_vector_store}'")

        # Extract the latest user message
        user_query = messages[-1].get("content", "") if messages else ""
        if not user_query or not user_query.strip():
            return "Error: User query is empty.", 0

        # Normalize strategy/retrieval labels -> canonical keys expected by core
        rag_key = RAG_STRATEGY_MAP.get(_norm(rag_method), DEFAULT_RAG_STRATEGY)
        retrieval_key = RETRIEVAL_METHOD_MAP.get(_norm(retrieval_method), DEFAULT_RETRIEVAL)
        
        logger.info(f"Normalized rag_key: '{rag_key}'")
        logger.info(f"Normalized retrieval_key: '{retrieval_key}'")

        # Create query engine for selected vector store (if local retrieval is used)
        active_query_engine = self.local_query_engine
        if selected_vector_store and retrieval_key in ["local", "hybrid", "automatic"]:
            logger.info(f"Creating query engine for selected vector store: {selected_vector_store}")
            active_query_engine = self.create_query_engine_for_vector_store(selected_vector_store)
        
        # Ensure agent is initialized before use
        self._ensure_agent_initialized()
        
        start_time = time.time()
        try:
            mcp_response = await process_model_context_query(
                query=user_query,
                context_memory=messages,
                tool_outputs=[],
                scratchpad="",
                agent_instance=self.agent,
                rac_corrector_instance=self.rac_corrector,
                testing_mode=self.testing_mode_enabled,
                suppress_threshold=0.4,
                flag_threshold=0.6,
                selected_rag_strategy=rag_key,
                selected_retrieval_method=retrieval_key,
                local_query_engine=active_query_engine,  # Use the selected query engine
                google_custom_search_instance=self.google_search_instance,
                tools_for_agent=self.tools_for_agent,
                skip_rac=skip_rac,
                selected_vector_store=selected_vector_store  # Pass to process_model_context_query
            )

            final_answer = mcp_response.get("final_answer", "No answer generated.")

            # Parse thinking and final answer from the response
            thinking_content, parsed_answer = parse_thinking_and_answer(final_answer)
            
            # If thinking was found, update final_answer to only contain the answer part
            if thinking_content:
                final_answer = parsed_answer
                logger.info(f"Separated thinking ({len(thinking_content)} chars) from answer ({len(final_answer)} chars)")
            
            # If the core complained about an invalid strategy (legacy UI strings, etc.),
            # retry once with safe defaults to avoid user-facing suppression.
            if final_answer.startswith("Invalid RAG strategy selected"):
                logger.warning("Invalid RAG strategy from upstream. Retrying with safe defaults (multi_step_query_engine, hybrid).")

                mcp_response = await process_model_context_query(
                    query=user_query,
                    context_memory=messages,
                    tool_outputs=[],
                    scratchpad="",
                    agent_instance=self.agent,
                    rac_corrector_instance=self.rac_corrector,
                    testing_mode=self.testing_mode_enabled,
                    suppress_threshold=0.4,
                    flag_threshold=0.6,
                    selected_rag_strategy=DEFAULT_RAG_STRATEGY,
                    selected_retrieval_method=DEFAULT_RETRIEVAL,
                    local_query_engine=self.local_query_engine,
                    google_custom_search_instance=self.google_search_instance,
                    tools_for_agent=self.tools_for_agent,
                    skip_rac=skip_rac
                )
                final_answer = mcp_response.get("final_answer", "No answer generated.")
                # Parse again after retry
                thinking_content, parsed_answer = parse_thinking_and_answer(final_answer)
                if thinking_content:
                    final_answer = parsed_answer

            sources_info = mcp_response.get("sources_used", {})
            
            # Extract image information first
            images = sources_info.get("images", [])
            image_info = None
            if images:
                # Format images for frontend - handle both Google images and local images
                formatted_images = []
                for img in images:
                    if isinstance(img, dict):
                        # Check if it's a Google Image (has 'link' key) or local image (has 'path'/'name' key)
                        if 'link' in img:
                            # Google Image from web search
                            formatted_images.append({
                                "path": img.get("link", ""),  # Full image URL
                                "caption": img.get("title", ""),
                                "page": "Web",
                                "source_file": img.get("context_link", "Google Images"),
                                "thumbnail": img.get("thumbnail", "")  # Add thumbnail URL
                            })
                        else:
                            # Local image from database
                            formatted_images.append({
                                "path": img.get("display_name", img.get("name", "")),
                                "caption": img.get("caption", ""),
                                "page": img.get("page", "N/A"),
                                "source_file": img.get("source_file", "Unknown")
                            })
                
                image_info = {
                    "images": formatted_images,
                    "count": len(formatted_images)
                }
            
            # Format sources (images will be handled separately by frontend)
            sources_str = self.format_sources_info(sources_info, include_images_in_text=False)
            formatted_response = f"{final_answer}\n\n{sources_str}"

            duration = int((time.time() - start_time) * 1000)
            return formatted_response, duration, image_info, thinking_content

        except Exception as e:
            logger.error(f"Error in ChatEngine.generate_response: {e}", exc_info=True)
            duration = int((time.time() - start_time) * 1000)
            return f"Error: An unexpected error occurred while processing your request. Details: {str(e)}", duration, None, None

    async def generate_response_with_progress(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        preset: str,
        temperature: float,
        user_id: str,
        rag_method: str,
        retrieval_method: str,
        selected_vector_store: Optional[str] = None,
        skip_rac: bool = False,
        progress_callback = None
    ) -> tuple[str, int, Optional[Dict[str, Any]]]:
        """
        Orchestrates the RAG pipeline with progress updates and returns (response_text, duration_ms, image_info).
        """
        logger.info(f"Received request with progress tracking for conversation_id: {conversation_id}")
        
        if progress_callback:
            await progress_callback("initializing", "Starting query processing...")
        
        logger.info(f"User ID: {user_id}")
        logger.info(f"RAW retrieval_method received: '{retrieval_method}'")
        logger.info(f"RAW rag_method received: '{rag_method}'")
        logger.info(f"Selected vector store: '{selected_vector_store}'")

        # Extract the latest user message
        user_query = messages[-1].get("content", "") if messages else ""
        if not user_query or not user_query.strip():
            return "Error: User query is empty.", 0

        # Normalize strategy/retrieval labels
        rag_key = RAG_STRATEGY_MAP.get(_norm(rag_method), DEFAULT_RAG_STRATEGY)
        retrieval_key = RETRIEVAL_METHOD_MAP.get(_norm(retrieval_method), DEFAULT_RETRIEVAL)
        
        logger.info(f"Normalized rag_key: '{rag_key}'")
        logger.info(f"Normalized retrieval_key: '{retrieval_key}'")

        # Create query engine for selected vector store
        if progress_callback:
            await progress_callback("loading_knowledge", f"Loading knowledge base: {selected_vector_store or 'default'}...")
        
        active_query_engine = self.local_query_engine
        if selected_vector_store and retrieval_key in ["local", "hybrid", "automatic"]:
            logger.info(f"Creating query engine for selected vector store: {selected_vector_store}")
            active_query_engine = self.create_query_engine_for_vector_store(selected_vector_store)
        
        # Ensure agent is initialized
        if progress_callback:
            await progress_callback("initializing_agent", "Initializing AI agent...")
        
        self._ensure_agent_initialized()
        
        start_time = time.time()
        try:
            if progress_callback:
                await progress_callback("processing_query", "Processing your question...")
            
            # Import the progress-aware version
            from .Hybrid_Bot import process_model_context_query_with_progress
            
            mcp_response = await process_model_context_query_with_progress(
                query=user_query,
                context_memory=messages,
                tool_outputs=[],
                scratchpad="",
                agent_instance=self.agent,
                rac_corrector_instance=self.rac_corrector,
                testing_mode=self.testing_mode_enabled,
                suppress_threshold=0.4,
                flag_threshold=0.6,
                selected_rag_strategy=rag_key,
                selected_retrieval_method=retrieval_key,
                local_query_engine=active_query_engine,
                google_custom_search_instance=self.google_search_instance,
                tools_for_agent=self.tools_for_agent,
                skip_rac=skip_rac,
                selected_vector_store=selected_vector_store,
                progress_callback=progress_callback
            )

            final_answer = mcp_response.get("final_answer", "No answer generated.")

            # Retry with safe defaults if needed
            if final_answer.startswith("Invalid RAG strategy selected"):
                logger.warning("Invalid RAG strategy from upstream. Retrying with safe defaults.")
                
                if progress_callback:
                    await progress_callback("retrying", "Retrying with default settings...")

                mcp_response = await process_model_context_query_with_progress(
                    query=user_query,
                    context_memory=messages,
                    tool_outputs=[],
                    scratchpad="",
                    agent_instance=self.agent,
                    rac_corrector_instance=self.rac_corrector,
                    testing_mode=self.testing_mode_enabled,
                    suppress_threshold=0.4,
                    flag_threshold=0.6,
                    selected_rag_strategy=DEFAULT_RAG_STRATEGY,
                    selected_retrieval_method=DEFAULT_RETRIEVAL,
                    local_query_engine=self.local_query_engine,
                    google_custom_search_instance=self.google_search_instance,
                    tools_for_agent=self.tools_for_agent,
                    skip_rac=skip_rac,
                    progress_callback=progress_callback
                )
                final_answer = mcp_response.get("final_answer", "No answer generated.")

            if progress_callback:
                await progress_callback("formatting_response", "Formatting response...")

            sources_info = mcp_response.get("sources_used", {})
            
            # Extract image information
            images = sources_info.get("images", [])
            image_info = None
            if images:
                formatted_images = []
                for img in images:
                    if isinstance(img, dict):
                        if 'link' in img:
                            # Google Image
                            formatted_images.append({
                                "path": img.get("link", ""),
                                "caption": img.get("title", ""),
                                "page": "Web",
                                "source_file": img.get("context_link", "Google Images"),
                                "thumbnail": img.get("thumbnail", "")
                            })
                        else:
                            # Local image
                            formatted_images.append({
                                "path": img.get("display_name", img.get("name", "")),
                                "caption": img.get("caption", ""),
                                "page": img.get("page", "N/A"),
                                "source_file": img.get("source_file", "Unknown")
                            })
                
                image_info = {
                    "images": formatted_images,
                    "count": len(formatted_images)
                }
            
            # Format sources
            sources_str = self.format_sources_info(sources_info, include_images_in_text=False)
            formatted_response = f"{final_answer}\n\n{sources_str}"

            if progress_callback:
                await progress_callback("complete", "Response ready!")

            duration = int((time.time() - start_time) * 1000)
            return formatted_response, duration, image_info

        except Exception as e:
            logger.error(f"Error in ChatEngine.generate_response_with_progress: {e}", exc_info=True)
            if progress_callback:
                await progress_callback("error", f"Error: {str(e)}")
            duration = int((time.time() - start_time) * 1000)
            return f"Error: An unexpected error occurred while processing your request. Details: {str(e)}", duration, None

    def format_sources_info(self, sources_info: Dict[str, Any], include_images_in_text: bool = True) -> str:
        """Formats sources into a user-friendly string."""
        info_lines = ["📚 **Sources Used:**"]
        local_files = sources_info.get('local_files', [])
        web_links = sources_info.get('web_links', [])
        images = sources_info.get('images', [])
        used_local = sources_info.get('used_local', False)
        used_web = sources_info.get('used_web', False)

        if local_files:
            info_lines.append("  📄 **Local PDF Documents Referenced:**")
            for i, filename in enumerate(local_files, 1):
                info_lines.append(f"    {i}. {filename}")
        elif used_local:
            # Local was queried but no specific files were found
            info_lines.append("  📄 **Local Knowledge Base Consulted:**")
            info_lines.append("    (No directly relevant documents found for this specific query)")

        # Only add image information to text if requested (for backward compatibility)
        # When include_images_in_text=False, images will be displayed separately by frontend
        if include_images_in_text:
            if images:
                info_lines.append(f"  🖼️ **Related Images ({len(images)}):**")
                for i, img in enumerate(images, 1):
                    caption = img.get('caption', 'No caption')
                    source_file = img.get('source_file', 'Unknown')
                    page = img.get('page', 'N/A')
                    info_lines.append(f"    {i}. {caption}")
                    info_lines.append(f"       Source: {source_file}, Page: {page}")
            elif used_local:
                # Local was used but no images found
                info_lines.append("  🖼️ **Related Images:**")
                info_lines.append("    No images found")

        if web_links:
            info_lines.append("  🌐 **Web Sources Referenced:**")
            for i, link in enumerate(web_links, 1):
                title = link.get('title', 'Unknown Title')
                url = link.get('url', '')
                info_lines.append(f"    {i}. {title} ({url})")
        elif used_web:
            # Web was queried but no specific links were found
            info_lines.append("  🌐 **Web Search Consulted:**")
            info_lines.append("    (No directly relevant web results found for this specific query)")

        if not local_files and not web_links and not images and not used_local and not used_web:
            info_lines.append("  ℹ️ No external sources were consulted.")

        return "\n".join(info_lines)


# -----------------------
# Standalone CLI (optional)
# -----------------------
if __name__ == "__main__":
    # Load env and spin up for quick manual testing
    load_dotenv()
    engine = ChatEngine()

    async def main_cli():
        while True:
            user_question = input("\nEnter your question (or 'exit'): ").strip()
            if user_question.lower() == 'exit':
                break

            print("--- Generating Response ---")

            mock_request_params = {
                "messages": [{"content": user_question, "sender": "user"}],
                "conversation_id": str(uuid.uuid4()),
                "model": "llama3",
                "preset": "default",
                "temperature": 0.7,
                "user_id": "test_user_id",
                "rag_method": "No Specific RAG Method",
                "retrieval_method": "local context only",
            }

            start_time = time.time()
            response_text, _duration_ms = await engine.generate_response(**mock_request_params)
            end_time = time.time()

            print("\n" + "=" * 50)
            print(f"Final Answer (Time: {end_time - start_time:.2f}s):")
            print("=" * 50)
            print(response_text)

    asyncio.run(main_cli())