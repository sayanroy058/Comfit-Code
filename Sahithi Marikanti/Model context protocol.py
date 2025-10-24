import os
import logging
import sys
import PyPDF2
import re
import json
import requests
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global Exception Hook for Tracebacks ---
def my_excepthook(type, value, traceback):
    import traceback as tb
    tb.print_exception(type, value, traceback)
    sys.exit(1)
sys.excepthook = my_excepthook

# --- Logging ---
# Basic logging for  application's INFO messages (e.g., "Starting Hybrid Chatbot...")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from LlamaIndex and HTTP client libraries
# Set them to WARNING, ERROR, or CRITICAL to hide most messages
logging.getLogger('llama_index').setLevel(logging.ERROR) # Make it more aggressive
logging.getLogger('httpx').setLevel(logging.WARNING) # For HTTP client requests
logging.getLogger('httpcore').setLevel(logging.WARNING) # For HTTP core requests
logging.getLogger('urllib3').setLevel(logging.WARNING) # Sometimes requests uses this

# DO NOT use set_global_handler("simple") if you want maximum suppression
# from llama_index.core import set_global_handler


from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import BaseModel, Field

# --- LLM Setup (Change model to 'tinyllama' if you want fast responses for debugging)
OPTIMAL_LLM_MODEL_NAME = "llama3"
OPTIMAL_LLM = Ollama(model=OPTIMAL_LLM_MODEL_NAME, request_timeout=600.0)
# Configure embedding model for LlamaIndex globally
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text") # Or another suitable embedding model
Settings.llm = OPTIMAL_LLM # Set the global LLM as well for consistency


# --- For Google Search ---
def validate_google_api_keys_from_env():
    """Validates the presence of Google API keys in environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key:
        logger.error("Error: GOOGLE_API_KEY environment variable is not set.")
        return None, None
    if not google_cse_id:
        logger.error("Error: GOOGLE_CSE_ID environment variable is not set.")
        return None, None
    logger.info("Google API Key and CSE ID environment variables validated.")
    return google_api_key, google_cse_id

class GoogleCustomSearchTool:
    """A tool to perform Google Custom Searches."""
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> str:
        """
        Performs a Google Custom Search and returns formatted results.
        Use this tool for questions that require up-to-date or broad web information.
        """
        logger.info(f"--- Google Search: Searching for '{query}' ---") # This logger.info will still appear
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": self.num_results
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            search_results = response.json()
            formatted_results = []
            if "items" in search_results:
                for i, item in enumerate(search_results["items"]):
                    formatted_results.append(
                        f"Result {i+1}: Title: {item.get('title', 'N/A')}\n"
                        f"Snippet: {item.get('snippet', 'N/A')}\n"
                        f"Link: {item.get('link', 'N/A')}\n"
                        f"---"
                    )
                return "\n".join(formatted_results)
            else:
                return "No relevant search results found for this query."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during Google Custom Search API call: {e}")
            return f"Error performing web search: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from Google Search API: {e}")
            return "Error processing web search results."

def clean_text(text):
    """Cleans text by removing hyphens, newlines, page numbers, and excessive whitespace."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # Handle hyphenated words broken by newline
    text = re.sub(r'[.!?]\n', '. ', text) # Replace period/exclamation/question mark followed by newline with period and space
    text = re.sub(r'[,;]\n', ', ', text) # Replace comma/semicolon followed by newline with comma and space
    text = text.replace('\n', ' ') # Replace all other newlines with spaces
    text = re.sub(r'\s{2,}', ' ', text).strip() # Replace multiple spaces with single space and strip leading/trailing whitespace
    text = re.sub(r'--- PAGE \d+ ---', '', text) # Remove page markers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # Remove lines containing only numbers (potential page numbers)
    text = text.strip() # Final strip
    return text

def curate_pdf_to_text(pdf_path, output_dir):
    """
    Extracts text from a PDF, cleans it, and saves it to a .txt file.
    Args:
        pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory to save the processed text file.
    Returns:
        str: The path to the saved text file, or None if processing failed.
    """
    txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
    output_filepath = os.path.join(output_dir, txt_filename)
    logger.info(f"Processing PDF: {os.path.basename(pdf_path)}...")
    full_text_pages = []
    try:
        with open(pdf_path, 'rb') as file:
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
            logger.warning(f"Extracted and cleaned text from '{pdf_path}' is empty. Skipping.")
            return None
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(final_curated_text)
        logger.info(f"Successfully curated and saved text to: {output_filepath}")
        return output_filepath
    except PyPDF2.errors.PdfReadError:
        logger.critical(f"FATAL ERROR: Could not read PDF file '{pdf_path}'. It might be encrypted or corrupted. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL ERROR: An unexpected error occurred while processing '{pdf_path}': {e}. Exiting.")
        sys.exit(1)

def load_single_document_for_indexing(file_path: str):
    """
    Loads a single text document using SimpleDirectoryReader for LlamaIndex indexing.
    Args:
        file_path (str): The path to the text file.
    Returns:
        list: A list of Document objects loaded from the file.
    """
    if not os.path.exists(file_path):
        logger.critical(f"FATAL ERROR: Processed text file '{file_path}' not found. Exiting.")
        sys.exit(1)
    logger.info(f"Loading document for indexing: {os.path.basename(file_path)}")
    reader = SimpleDirectoryReader(input_files=[file_path], required_exts=[".txt"])
    documents = reader.load_data()
    if not documents:
        logger.critical(f"FATAL ERROR: No content could be loaded from '{file_path}' for indexing. Exiting.")
        sys.exit(1)
    filename = os.path.basename(file_path)
    dummy_category = "BookContent" # Placeholder category
    for doc in documents:
        doc.metadata['category'] = dummy_category # it is just a tag if u place any other doc you can chanhge the tag name
        doc.metadata['filename'] = filename
    logger.info(f"Loaded {len(documents)} document segments for indexing from '{filename}'.")
    return documents

def main():
    """Main function to run the Hybrid Chatbot with MCP and LlamaIndex Agents."""
    logger.info("Starting Hybrid Chatbot (Model Context Protocol with LlamaIndex Agents)...")

    if len(sys.argv) < 2:
        logger.critical("FATAL ERROR: Please provide the path to  PDF book as a command-line argument.")
        logger.critical("Example: python mcp.py \"./ebooks/Product Fit and Sizing_25_06_03_12_44_21.pdf\"")
        sys.exit(1)
    pdf_file_path = sys.argv[1]

    CURATED_DATA_SINGLE_BOOK_DIR = 'curated_data_single_book'
    os.makedirs(CURATED_DATA_SINGLE_BOOK_DIR, exist_ok=True)

    # Process the PDF to a text file
    processed_text_file_path = curate_pdf_to_text(pdf_file_path, CURATED_DATA_SINGLE_BOOK_DIR)
    if not processed_text_file_path:
        logger.critical(f"FATAL ERROR: Could not process PDF '{pdf_file_path}'. Exiting.")
        sys.exit(1)

    # --- Step 1: Load and Index Local Document ---
    documents = load_single_document_for_indexing(processed_text_file_path)
    logger.info("Creating VectorStoreIndex for local PDF data...")
    try:
        # Explicitly pass the LLM and Embedding Model to the index
        # This ensures Ollama models are used and prevents OpenAI fallback
        local_index = VectorStoreIndex.from_documents(
            documents,
            llm=OPTIMAL_LLM,
            embed_model=Settings.embed_model,
            # transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)], # Example custom chunking
        )
        local_query_engine = local_index.as_query_engine(llm=OPTIMAL_LLM) # Ensure query engine also uses Ollama
        logger.info("Local PDF data indexed successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex for local data: {e}. Exiting.")
        sys.exit(1)

    # --- Step 2: Initialize Google Search Tool ---
    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. Cannot perform web searches. Exiting.")
        sys.exit(1)
    
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5
    )

    # --- Step 3: Define Tools for the Agent (MCP Servers) ---
# --- Step 3: Define Tools for the Agent (MCP Servers) ---

    # Define a Pydantic model for the local_book_qa_function's input
    class LocalBookQAToolInput(BaseModel):
        query: str = Field(description="The question to ask about the PDF book content.")

    # Define a helper function that explicitly calls  local query engine
    def local_book_qa_function(query: str) -> str:
        """
        Useful for questions specifically about the content of the provided PDF book.
        Use this tool when the user's question relates to information likely found
        within the PDF document.
        """
        logger.info(f"--- Local RAG: Querying for '{query}' ---") # This INFO log will still appear
        response = local_query_engine.query(query) # Call the query engine directly with the string query
        return str(response) # Return the string representation of the response

    # Tool for local PDF RAG - Now using FunctionTool with our new helper function and explicit input schema
    local_rag_tool = FunctionTool.from_defaults(
        fn=local_book_qa_function,
        name="local_book_qa",
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "Use this tool when the user's question relates to information likely found "
            "within the PDF document."
        ),
        # Explicitly define the input schema for the tool
        fn_schema=LocalBookQAToolInput,
    )

    # Tool for Google Search (This one stays the same, it likely already has a string-based schema)
    Google_Search_tool_for_agent = FunctionTool.from_defaults(
        fn= Google_Search_instance.search,
        name="google_web_search",
        description=(
            "Useful for answering general knowledge questions, current events, or anything "
            "that requires searching the internet. Always consider using this for questions "
            "that are not likely to be in the provided PDF book, or to verify facts."
        ),
    )

    # Make sure this list contains  newly defined tools
    tools = [local_rag_tool, Google_Search_tool_for_agent]
    # --- Step 4: Initialize the ReAct Agent (MCP Host/Client) ---
    logger.info(f"Initializing ReAct Agent with LLM: {OPTIMAL_LLM_MODEL_NAME}...")
    # Settings.llm is already set globally, but explicitly passing to agent is good practice
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=OPTIMAL_LLM,
        verbose=False, # Changed to False for cleaner output
        max_iterations=10 # Increased for more robust operation
    )

    logger.info("ReAct Agent initialized.")

    logger.info("\n--- Hybrid Chatbot (MCP with LlamaIndex Agents) READY ---")
    logger.info(f"Agent uses LLM: {OPTIMAL_LLM_MODEL_NAME}")
    logger.info(f"Tools available: local_book_qa, google_web_search")
    logger.info(f"Type  questions. Type 'exit' to quit.")

    # --- Step 5: Interactive Query Loop ---
    while True:
        user_question = input("\nEnter  question (or 'exit'): ").strip()
        if user_question.lower() == 'exit':
            logger.info("Exiting Hybrid Chatbot. Goodbye!")
            break

        print("\n" + "="*50)
        print("--- Agent's Response ---")
        print("="*50)
        try:
            response = agent.chat(user_question)
            # This will print the final answer string from the agent
            # The verbose output from the LLM's internal "thought" process
            # is now mostly suppressed by the logging level settings.
            print(response.response) 
        except Exception as e:
            logger.error(f"Error during agent interaction: {e}")
            import traceback
            traceback.print_exc()
            print("An error occurred while processing  request.")
        print("="*50 + "\n")

    logger.info("Hybrid Chatbot Session Completed.")

if __name__ == "__main__":
    main()
