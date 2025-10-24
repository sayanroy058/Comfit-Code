import os
import logging
import sys
import PyPDF2
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
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
from pydantic import BaseModel, Field

# --- Model Context Protocol Prompt Template ---
# This is now primarily for the agent's internal system prompt or other specific LLM calls,
# not for the final response synthesis step.
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

# --- LLM Setup ---
OPTIMAL_LLM_MODEL_NAME = "llama3"
OPTIMAL_LLM = Ollama(model=OPTIMAL_LLM_MODEL_NAME, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = OPTIMAL_LLM

# --- RAC (Retrieval-Augmented Correction) Implementation ---
class FactualClaimExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self, text: str) -> List[str]:
        logger.info("Extracting factual claims from the response...")
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
                    if claim and len(claim) > 10:
                        claims.append(claim)
            logger.info(f"Extracted {len(claims)} factual claims")
            return claims
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

class FactVerifier:
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.local_query_engine = local_query_engine
        self.Google_Search_tool = Google_Search_tool
        self.retrieval_method = retrieval_method
        self.verification_cache = {}
        self.reversal_min_confidence = 0.95

    def _get_claim_hash(self, claim: str) -> str:
        return hashlib.md5(claim.lower().encode('utf-8')).hexdigest()

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        claim_hash = self._get_claim_hash(claim)
        if claim_hash in self.verification_cache:
            logger.info(f"Cache hit for claim: {claim[:50]}...")
            return self.verification_cache[claim_hash]
        
        logger.info(f"Verifying claim: {claim[:50]}... with retrieval method: {self.retrieval_method}")
        evidence = []
        
        use_local_source, use_web_source = False, False

        # Apply the selected retrieval method logic
        if self.retrieval_method == "local":
            use_local_source = True
        elif self.retrieval_method == "web":
            use_web_source = True
        elif self.retrieval_method == "hybrid":
            use_local_source = True
            use_web_source = True
        elif self.retrieval_method == "automatic":
            # Simple heuristic for automatic mode
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
                        local_result = self.local_query_engine.query(query)
                        local_content = str(local_result).strip()
                        if (local_content and len(local_content) > 20 and
                            not any(phrase in local_content.lower() for phrase in [
                                "i don't know", "no information", "not mentioned",
                                "cannot find", "not available", "no details"])):
                            evidence.append({
                                'source': 'local_knowledge',
                                'content': local_content,
                                'confidence': 0.9,
                                'query_used': query
                            })
                            local_evidence_found = True
                            logger.info(f"Local evidence found for query: {query[:50]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Local query failed for '{query[:30]}...': {e}")
                        continue
                if not local_evidence_found:
                    logger.info("No relevant local evidence found.")
            except Exception as e:
                logger.warning(f"Local verification failed: {e}")
        
        if use_web_source and not evidence:  # Only use web if no local evidence found or if local is not chosen
            try:
                logger.info(f"Searching web for: {claim[:50]}...")
                search_query = self._extract_search_terms(claim)
                web_result = self.Google_Search_tool.search(search_query)
                if web_result and "No relevant search results" not in web_result:
                    evidence.append({
                        'source': 'web_search',
                        'content': web_result,
                        'confidence': 0.7,
                        'query_used': search_query
                    })
                    logger.info(f"Web evidence found for query: {search_query}")
            except Exception as e:
                logger.warning(f"Web verification failed: {e}")
        
        evidence_sources = [e['source'] for e in evidence]
        logger.info(f"Evidence sources used: {evidence_sources}")
        verification_result = self._analyze_evidence(claim, evidence)
        
        result_to_cache = {
            'claim': claim,
            'is_supported': verification_result['is_supported'],
            'confidence': verification_result['confidence'],
            'evidence': evidence,
            'correction_suggestion': verification_result.get('correction', None),
            'warning': verification_result.get('warning', None)
        }
        self.verification_cache[claim_hash] = result_to_cache
        return result_to_cache
    
    def _extract_search_terms(self, claim: str) -> str:
        words = claim.split()
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        return ' '.join(key_words[:5])
    
    def _analyze_evidence(self, claim: str, evidence: List[Dict]) -> Dict[str, Any]:
        if not evidence:
            return {
                'is_supported': False,
                'confidence': 0.0,
                'correction': None,
                'warning': "No evidence found to verify this claim."
            }
        
        local_evidence = [e for e in evidence if e['source'] == 'local_knowledge']
        web_evidence = [e for e in evidence if e['source'] == 'web_search']
        
        evidence_text = ""
        if local_evidence:
            evidence_text += "=== LOCAL DOCUMENT EVIDENCE ===\n"
            for e in local_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        if web_evidence:
            evidence_text += "=== WEB SEARCH EVIDENCE ===\n"
            for e in web_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        
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
        5. For reversals, require confidence >= {self.reversal_min_confidence}.
        Respond:
        VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE]
        CONFIDENCE: [0.0-1.0]
        CORRECTION: [If contradicted, provide correction, else "None"]
        REASONING: [Brief explanation]
        """
        
        try:
            response = str(self.llm.complete(analysis_prompt))
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
                        confidence = 0.5
                elif line.startswith('CORRECTION:'):
                    correction_text = line.replace('CORRECTION:', '').strip()
                    if correction_text.lower() != 'none':
                        correction = correction_text
            
            if local_evidence and verdict in ["SUPPORTED", "CONTRADICTED"]:
                confidence = min(confidence + 0.2, 1.0)
            
            is_supported = verdict == "SUPPORTED"
            negative_markers = ["not", "no", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "never", "false", "untrue", "without"]
            original_has_neg = any(neg_m in claim.lower() for neg_m in negative_markers)
            correction_has_neg = any(neg_m in (correction or "").lower() for neg_m in negative_markers)
            
            if correction and ((original_has_neg and not correction_has_neg) or (not original_has_neg and correction_has_neg)):
                if confidence < self.reversal_min_confidence:
                    logger.warning(f"Potential reversal detected but confidence ({confidence:.2f}) below threshold.")
                    correction = None
                    verdict = "INSUFFICIENT_EVIDENCE"
                    confidence = 0.5
            
            warning_message = None
            if not is_supported and confidence <= 0.6:
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
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.claim_extractor = FactualClaimExtractor(llm)
        # Pass the retrieval method to the FactVerifier
        self.fact_verifier = FactVerifier(llm, local_query_engine, Google_Search_tool, retrieval_method)
        self.retrieval_method = retrieval_method
        self.correction_threshold = 0.5
        self.uncertainty_threshold = 0.6
        self.local_priority = True
        self.testing_mode = False
        self.rac_enabled = True

    def correct_response(self, original_response: str, apply_corrections: bool = True) -> Dict[str, Any]:
        if not self.rac_enabled:
            logger.info("RAC is disabled, skipping correction.")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 1.0
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
                'average_confidence': 0.0
            }
            
        verification_results = []
        corrections_needed = []
        uncertain_claims = []
        
        # Ensure the FactVerifier instance has the correct retrieval mode
        self.fact_verifier.retrieval_method = self.retrieval_method
        
        start_verification = time.perf_counter()
        for i, claim in enumerate(claims, 1):
            logger.info(f"Processing claim {i}/{len(claims)}: {claim[:50]}...")
            start_single_verify = time.perf_counter()
            result = self.fact_verifier.verify_claim(claim) # Uses the mode set above
            end_single_verify = time.perf_counter()
            logger.info(f"Timing - Single Claim Verification ({i}): {end_single_verify - start_single_verify:.4f} seconds")
            
            verification_results.append(result)
            evidence_sources = [e['source'] for e in result['evidence']]
            logger.info(f"Claim {i} verification: {result['is_supported']}, confidence: {result['confidence']:.2f}, sources: {evidence_sources}")
            
            if not result['is_supported'] and result['confidence'] > self.correction_threshold:
                if result['correction_suggestion']:
                    corrections_needed.append({
                        'original_claim': claim,
                        'correction': result['correction_suggestion'],
                        'confidence': result['confidence'],
                        'evidence_sources': evidence_sources
                    })
                    logger.info(f"Correction needed for claim {i}")
                else:
                    logger.info(f"Claim {i} not supported but no correction available")
            
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
        if apply_corrections and not self.testing_mode and corrections_needed:
            start_apply_corrections = time.perf_counter()
            corrected_response = self._apply_corrections(original_response, corrections_needed)
            end_apply_corrections = time.perf_counter()
            logger.info(f"Timing - Applying Corrections: {end_apply_corrections - start_apply_corrections:.4f} seconds")
            
        logger.info(f"RAC correction completed. Analyzed {len(claims)} claims, made {len(corrections_needed)} corrections")
        
        total_confidence = sum(res['confidence'] for res in verification_results)
        average_confidence = total_confidence / len(claims) if claims else 0.0
        
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
        correction_prompt = f"""
        Task: Apply corrections to the original response while maintaining its structure and flow.
        ORIGINAL RESPONSE:
        {original_response}
        CORRECTIONS TO APPLY:
        {chr(10).join([f"- Replace/correct: '{c['original_claim']}' → '{c['correction']}'" for c in corrections])}
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
            return original_response

# --- Google Search Tool ---
def validate_google_api_keys_from_env():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        logger.error("Google API keys not configured.")
        return None, None
    logger.info("Google API keys validated.")
    return google_api_key, google_cse_id

class GoogleCustomSearchTool:
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> str:
        logger.info(f"Google Search: '{query}'")
        start_web_api = time.perf_counter()
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
                return "No relevant search results found."
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API error: {e}")
            return f"Error performing web search: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return "Error processing web search results."
        finally:
            end_web_api = time.perf_counter()
            logger.info(f"Timing - Google Web Search API Call: {end_web_api - start_web_api:.4f} seconds")

# --- PDF Processing Functions ---
def clean_text(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[.!?]\n', '. ', text)
    text = re.sub(r'[,;]\n', ', ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def curate_pdf_to_text(pdf_path_str, output_dir):
    pdf_path = Path(pdf_path_str)
    
    if not pdf_path.is_file():
        logger.critical(f"FATAL ERROR: PDF file not found at '{pdf_path_str}'. Exiting.")
        sys.exit(1)

    temp_dir = tempfile.mkdtemp()
    sanitized_filename = "temp_pdf.pdf"
    temp_pdf_path = Path(temp_dir) / sanitized_filename

    try:
        shutil.copy(pdf_path, temp_pdf_path)
        logger.info(f"Copied original PDF to temporary path: {temp_pdf_path}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not copy PDF file from '{pdf_path_str}' to temporary location. Error: {e}. Exiting.")
        shutil.rmtree(temp_dir)
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
        logger.critical(f"FATAL ERROR: Could not read PDF '{temp_pdf_path}'. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL ERROR: Error processing '{temp_pdf_path}': {e}. Exiting.")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir)

def load_single_document_for_indexing(file_path: str):
    if not os.path.exists(file_path):
        logger.critical(f"FATAL ERROR: Text file '{file_path}' not found. Exiting.")
        sys.exit(1)
    logger.info(f"Loading document for indexing: {os.path.basename(file_path)}")
    reader = SimpleDirectoryReader(input_files=[file_path], required_exts=[".txt"])
    documents = reader.load_data()
    if not documents:
        logger.critical(f"FATAL ERROR: No content loaded from '{file_path}'. Exiting.")
        sys.exit(1)
    filename = os.path.basename(file_path)
    dummy_category = "BookContent"
    for doc in documents:
        doc.metadata['category'] = dummy_category
        doc.metadata['filename'] = filename
    logger.info(f"Loaded {len(documents)} document segments from '{filename}'.")
    return documents

# --- RAG Strategy Implementations ---
async def run_planning_workflow(query: str, agent_instance: ReActAgent, trace: List[str]) -> str:
    trace.append(f"Strategy: Planning Workflow - Agent thinking on '{query}'...")
    try:
        response_obj = await asyncio.to_thread(agent_instance.chat, query)
        response = response_obj.response
        trace.append(f"Planning Workflow Raw Response: {response}")
        return response
    except Exception as e:
        trace.append(f"Error in Planning Workflow: {e}")
        logger.error(f"Error running planning workflow: {e}", exc_info=True)
        return "An error occurred during the planning workflow."

async def run_multi_step_query_engine_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> str:
    trace.append(f"Strategy: Multi-Step Query Engine - Routing '{query}'...")
    from llama_index.core.base.response.schema import Response
    
    local_tool_choice = QueryEngineTool.from_defaults(
        query_engine=local_query_engine,
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "Use when the question relates to 'speed process', 'anthropometry', 'product fit', 'sizing', etc."
        ),
    )
    
    class GoogleQueryEngine:
        def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
            self.search_tool = search_tool_instance
            self.llm = llm
        
        async def aquery(self, query_str: str) -> Response:
            raw_search_result = await asyncio.to_thread(self.search_tool.search, query_str)
            
            if "No relevant search results" in raw_search_result:
                synthesized_answer = "No relevant information found on the web."
                return Response(response=synthesized_answer, metadata={"source": "Google Search"})

            synthesis_prompt = f"""
            Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
            Web Search Results:
            {raw_search_result}
            If the results do not contain a clear answer, state that.
            Provide only the answer, without referring to the search process or tools used.
            """
            try:
                synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                synthesized_answer = str(synthesized_answer)
            except Exception as e:
                logger.error(f"Error during Google search synthesis: {e}")
                synthesized_answer = "Could not synthesize an answer from web search results."
            return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis"})

        def query(self, query_str: str) -> Response:
            return asyncio.run(self.aquery(query_str))

    google_qe_instance = GoogleQueryEngine(google_custom_search_instance, OPTIMAL_LLM)
    
    query_engine_tools = []
    # Use the tools_for_agent list to dynamically decide which query engines to add
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
        return "No relevant query engine available for the selected retrieval method."

    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=query_engine_tools,
        llm=OPTIMAL_LLM
    )
    
    try:
        response_obj = await asyncio.to_thread(router_query_engine.query, query)
        response = str(response_obj)
        trace.append(f"Multi-Step Query Engine Raw Response: {response}")
        return response
    except Exception as e:
        trace.append(f"Error in Multi-Step Query Engine Workflow: {e}")
        logger.error(f"Error running multi_step_query_engine_workflow: {e}", exc_info=True)
        return "An error occurred during the multi-step query engine workflow."

async def run_multi_strategy_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> str:
    trace.append(f"Strategy: Multi-Strategy Workflow - Executing multiple queries for '{query}'...")
    responses = []
    from llama_index.core.base.response.schema import Response
    
    use_local_source = any(tool.metadata.name == "local_book_qa" for tool in tools_for_agent)
    use_web_source = any(tool.metadata.name == "google_web_search" for tool in tools_for_agent)
    
    if use_local_source:
        try:
            local_response_obj = await asyncio.to_thread(local_query_engine.query, query)
            local_response = str(local_response_obj)
            responses.append(f"Local RAG result: {local_response}")
            trace.append(f"Multi-Strategy: Local RAG executed. Response snippet: {local_response[:100]}...")
        except Exception as e:
            responses.append(f"Local RAG error: {e}")
            trace.append(f"Multi-Strategy: Local RAG error: {e}")
            logger.warning(f"Error in Multi-Strategy local RAG: {e}")
    
    if use_web_source:
        class GoogleQueryEngineForMultiStrategy:
            def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
                self.search_tool = search_tool_instance
                self.llm = llm
            async def aquery(self, query_str: str) -> Response:
                raw_search_result = await asyncio.to_thread(self.search_tool.search, query_str)
                if "No relevant search results" in raw_search_result:
                    synthesized_answer = "No relevant information found on the web."
                    return Response(response=synthesized_answer, metadata={"source": "Google Search"})
                synthesis_prompt = f"""
                Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
                Web Search Results:
                {raw_search_result}
                If the results do not contain a clear answer, state that.
                Provide only the answer, without referring to the search process or tools used.
                """
                try:
                    synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                    synthesized_answer = str(synthesized_answer)
                except Exception as e:
                    logger.error(f"Error during Google search synthesis (MultiStrategy): {e}")
                    synthesized_answer = "Could not synthesize an answer from web search results."
                return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis"})
            def query(self, query_str: str) -> Response:
                return asyncio.run(self.aquery(query_str))
        
        google_qe_instance = GoogleQueryEngineForMultiStrategy(google_custom_search_instance, OPTIMAL_LLM)
        try:
            web_response_obj = await asyncio.to_thread(google_qe_instance.query, query)
            web_response = str(web_response_obj)
            responses.append(f"Web Search result: {web_response}")
            trace.append(f"Multi-Strategy: Web Search executed. Response snippet: {web_response[:100]}...")
        except Exception as e:
            responses.append(f"Web Search error: {e}")
            trace.append(f"Multi-Strategy: Web Search error: {e}")
            logger.warning(f"Error in Multi-Strategy web search: {e}")
    
    combined_info = "\n\n".join(responses)
    if not combined_info.strip():
        return "No information found from any strategy."
    
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
        return final_answer
    except Exception as e:
        trace.append(f"Error in Multi-Strategy Synthesis: {e}")
        logger.error(f"Error in multi-strategy synthesis: {e}", exc_info=True)
        return "An error occurred during multi-strategy synthesis."

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
    tools_for_agent: List[FunctionTool]
) -> Dict[str, Any]:
    """
    Processes a user query using a selected RAG strategy and applies RAC if enabled.
    This function orchestrates the entire response generation pipeline.
    """
    logger.info(f"Processing Model Context Query: '{query}' with strategy: {selected_rag_strategy}, retrieval: {selected_retrieval_method}")
    response_trace = [f"ModelContextQuery received: Query='{query}'"]
    response_trace.append(f"Selected RAG Strategy: {selected_rag_strategy}")
    response_trace.append(f"Selected Retrieval Method: {selected_retrieval_method}")

    start_total_process_mcp = time.perf_counter()
    try:
        start_preprocess = time.perf_counter()
        processed_question = query
        response_trace.append(f"Pre-processed query: '{processed_question}'")
        end_preprocess = time.perf_counter()
        response_trace.append(f"Timing - Preprocessing: {end_preprocess - start_preprocess:.4f} seconds")

        start_rag_strategy = time.perf_counter()
        original_response_text = ""
        tool_outputs = []
        scratchpad = ""
        
        # Create an agent instance for this query with the dynamically selected tools
        agent_instance_for_query = ReActAgent.from_tools(
            tools=tools_for_agent,
            llm=OPTIMAL_LLM,
            verbose=False,
            max_iterations=30
        )

        if selected_rag_strategy == "planning_workflow" or selected_rag_strategy == "rac_enhanced_hybrid_rag":
            original_response_text = await run_planning_workflow(processed_question, agent_instance_for_query, response_trace)
            tool_outputs.append({"tool": "planning_workflow", "result": original_response_text})
        elif selected_rag_strategy == "multi_step_query_engine":
            original_response_text = await run_multi_step_query_engine_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, tools_for_agent
            )
            tool_outputs.append({"tool": "multi_step_query_engine", "result": original_response_text})
        elif selected_rag_strategy == "multi_strategy_workflow":
            original_response_text = await run_multi_strategy_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, tools_for_agent
            )
            tool_outputs.append({"tool": "multi_strategy_workflow", "result": original_response_text})
        elif selected_rag_strategy == "no_method":
            agent_response_obj = await asyncio.to_thread(agent_instance_for_query.chat, processed_question)
            original_response_text = agent_response_obj.response
            tool_outputs.append({"tool": "react_agent", "result": original_response_text})
            response_trace.append(f"Agent raw response: '{original_response_text}'")
        else:
            original_response_text = "Invalid RAG strategy selected."
            logger.error(original_response_text)

        end_rag_strategy = time.perf_counter()
        response_trace.append(f"Timing - RAG Strategy ({selected_rag_strategy}) Execution: {end_rag_strategy - start_rag_strategy:.4f} seconds")

        final_answer_content = original_response_text
        average_conf = 1.0

        if rac_corrector_instance.rac_enabled:
            start_rac = time.perf_counter()
            response_trace.append("Applying RAC (Retrieval-Augmented Correction)...")
            
            # Set the retrieval method on the RAC corrector instance before running
            rac_corrector_instance.retrieval_method = selected_retrieval_method
            
            rac_result = rac_corrector_instance.correct_response(original_response_text, apply_corrections=not testing_mode)
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
            if average_conf < suppress_threshold:
                final_answer_content = f"❌ Response suppressed due to very low confidence ({average_conf:.2f})."
                response_trace.append(f"Confidence Cascade: Response Suppressed (Avg Confidence: {average_conf:.2f})")
            elif average_conf < flag_threshold:
                final_answer_content = f"⚠️ Low confidence in response ({average_conf:.2f}). Please use with caution.\n\n" + final_answer_content
                response_trace.append(f"Confidence Cascade: Response Flagged (Avg Confidence: {average_conf:.2f})")
            else:
                response_trace.append(f"Confidence Cascade: Response Accepted (Avg Confidence: {average_conf:.2f})")

        # --- IMPORTANT CHANGE ---
        # The final LLM synthesis step using MODEL_CONTEXT_PROMPT is removed.
        # The `final_answer_content` is already the user-ready response.
        # We now simply return it directly.
        final_answer = final_answer_content
        # End of change
        
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration: {end_total_process_mcp - start_total_process_mcp:.4f} seconds")

        return {
            "final_answer": final_answer,
            "trace": response_trace,
            "confidence_score": average_conf
        }
    except Exception as e:
        logger.error(f"Error in process_model_context_query: {e}", exc_info=True)
        error_message = "An unexpected error occurred while processing your request."
        response_trace.append(f"ERROR: {e}")
        response_trace.append(traceback.format_exc())
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration (Error): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")
        return {
            "final_answer": error_message,
            "trace": response_trace,
            "confidence_score": 0.0
        }

def main():
    logger.info("Starting Enhanced Hybrid Chatbot with Model Context Protocol...")
    if len(sys.argv) < 2:
        logger.critical("FATAL ERROR: Please provide the path to your PDF book as a command-line argument.")
        sys.exit(1)
    
    pdf_file_path = sys.argv[1]
    testing_mode_enabled = "--dry-run" in sys.argv
    if testing_mode_enabled:
        logger.info("RAC Testing Mode (--dry-run) enabled.")
        sys.argv.remove("--dry-run")

    CURATED_DATA_SINGLE_BOOK_DIR = 'curated_data_single_book'
    os.makedirs(CURATED_DATA_SINGLE_BOOK_DIR, exist_ok=True)
    
    processed_text_file_path = curate_pdf_to_text(pdf_file_path, CURATED_DATA_SINGLE_BOOK_DIR)
    if not processed_text_file_path:
        logger.critical(f"FATAL ERROR: Could not process PDF '{pdf_file_path}'. Exiting.")
        sys.exit(1)
    
    documents = load_single_document_for_indexing(processed_text_file_path)
    logger.info("Creating VectorStoreIndex for local PDF data...")
    try:
        local_index = VectorStoreIndex.from_documents(
            documents,
            llm=OPTIMAL_LLM,
            embed_model=Settings.embed_model,
        )
        local_query_engine = local_index.as_query_engine(llm=OPTIMAL_LLM)
        logger.info("Local PDF data indexed successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex: {e}. Exiting.")
        sys.exit(1)

    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. Exiting.")
        sys.exit(1)
    
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5
    )

    rac_corrector = RACCorrector(
        llm=OPTIMAL_LLM,
        local_query_engine=local_query_engine,
        Google_Search_tool=Google_Search_instance
    )
    rac_corrector.testing_mode = testing_mode_enabled
    
    class LocalBookQAToolInput(BaseModel):
        query: str = Field(description="The question to ask about the PDF book content.")

    def local_book_qa_function(query: str) -> str:
        logger.info(f"Local RAG: Querying for '{query}'")
        start_local_rag_query = time.perf_counter()
        response = local_query_engine.query(query)
        end_local_rag_query = time.time()
        logger.info(f"Timing - Local RAG Query: {end_local_rag_query - start_local_rag_query:.4f} seconds")
        return str(response)

    local_rag_tool = FunctionTool.from_defaults(
        fn=local_book_qa_function,
        name="local_book_qa",
        description=(
            "Useful for questions specifically about the content of the provided PDF book."
        ),
        fn_schema=LocalBookQAToolInput,
    )

    Google_Search_tool_for_agent = FunctionTool.from_defaults(
        fn=Google_Search_instance.search,
        name="google_web_search",
        description=(
            "Useful for general knowledge questions, current events, or anything requiring internet search."
        ),
    )

    tools_for_agent = [local_rag_tool, Google_Search_tool_for_agent]

    logger.info(f"Initializing ReAct Agent with LLM: {OPTIMAL_LLM_MODEL_NAME}...")
    # NOTE: The ReActAgent.from_tools method now handles setting up the prompt internally.
    agent = ReActAgent.from_tools(
        tools=tools_for_agent,
        llm=OPTIMAL_LLM,
        verbose=False,
        max_iterations=30
    )
    
    logger.info("Initialized Enhanced ReAct Agent with RAC.")
    logger.info("\n--- Enhanced Hybrid Chatbot with Model Context Protocol READY ---")
    logger.info(f"Agent uses LLM: {OPTIMAL_LLM_MODEL_NAME}")
    logger.info(f"Tools available: local_book_qa, google_web_search")
    logger.info(f"RAC enabled ({'Testing Mode' if testing_mode_enabled else 'Active'})")
    logger.info(f"Type your questions. Type 'exit' to quit.")
    logger.info(f"Commands: 'toggle_rac', 'rac_stats', 'set_mode [local|web|hybrid|automatic]'")
    
    rac_corrector.rac_enabled = True
    rac_stats = {
        'total_queries': 0,
        'corrected_queries': 0,
        'total_corrections': 0,
        'uncertain_claims_flagged': 0,
        'responses_suppressed': 0,
        'responses_flagged_low_confidence': 0
    }
    
    SUPPRESS_THRESHOLD = 0.4
    FLAG_THRESHOLD = 0.6

    RAG_STRATEGIES = {
        "1": "rac_enhanced_hybrid_rag",
        "2": "planning_workflow",
        "3": "multi_step_query_engine",
        "4": "multi_strategy_workflow",
        "5": "no_method"
    }
    STRATEGY_NAMES = {
        "rac_enhanced_hybrid_rag": "RAC Enhanced Hybrid RAG",
        "planning_workflow": "Planning Workflow",
        "multi_step_query_engine": "Multi-Step Query Engine",
        "multi_strategy_workflow": "Multi-Strategy Workflow",
        "no_method": "No Specific RAG Method"
    }
    current_rag_strategy = "1"

    RETRIEVAL_METHODS = {
        "local": "Local Only (PDF)",
        "web": "Web Only (Google Search)",
        "hybrid": "Local and Web",
        "automatic": "Automatic"
    }
    current_retrieval_method = "hybrid"
    
    context_memory = {}
    
    while True:
        print("\n" + "="*50)
        print("Select RAG Strategy:")
        for key, value in RAG_STRATEGIES.items():
            print(f"  {key}. {STRATEGY_NAMES[value]}")
        strategy_choice = input(f"Enter strategy number (currently using {STRATEGY_NAMES[RAG_STRATEGIES[current_rag_strategy]]}): ").strip().lower()
        
        if strategy_choice == 'exit':
            logger.info("Exiting Enhanced Hybrid Chatbot. Goodbye!")
            break
        elif strategy_choice == 'toggle_rac':
            rac_corrector.rac_enabled = not rac_corrector.rac_enabled
            print(f"RAC is now {'ENABLED' if rac_corrector.rac_enabled else 'DISABLED'}")
            continue
        elif strategy_choice == 'rac_stats':
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
            parts = strategy_choice.split()
            if len(parts) == 2 and parts[0] == 'set_mode':
                mode = parts[1]
                if mode in RETRIEVAL_METHODS:
                    current_retrieval_method = mode
                    print(f"Retrieval method set to: {mode.upper()}.")
                else:
                    print("Invalid mode. Use 'set_mode [local|web|hybrid|automatic]'.")
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

        tools_for_query = []
        if current_retrieval_method in ["local", "hybrid", "automatic"]:
            tools_for_query.append(local_rag_tool)
        if current_retrieval_method in ["web", "hybrid", "automatic"]:
            tools_for_query.append(Google_Search_tool_for_agent)
        
        if not tools_for_query:
            print("❌ No retrieval source selected. Please choose a method that includes a data source.")
            continue

        try:
            start_time = time.time()
            mcp_response = asyncio.run(process_model_context_query(
                query=user_question,
                context_memory=context_memory,
                tool_outputs=[],
                scratchpad="",
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=testing_mode_enabled,
                suppress_threshold=SUPPRESS_THRESHOLD,
                flag_threshold=FLAG_THRESHOLD,
                selected_rag_strategy=RAG_STRATEGIES[current_rag_strategy],
                selected_retrieval_method=current_retrieval_method,
                local_query_engine=local_query_engine,
                google_custom_search_instance=Google_Search_instance,
                tools_for_agent=tools_for_query
            ))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
            
            rac_stats['total_queries'] += 1
            if rac_corrector.rac_enabled:
                if "❌" in mcp_response["final_answer"]:
                    rac_stats['responses_suppressed'] += 1
                elif "⚠️" in mcp_response["final_answer"]:
                    rac_stats['responses_flagged_low_confidence'] += 1
                
            print(mcp_response["final_answer"])
            print("\n--- Model Context Trace ---")
            for step in mcp_response["trace"]:
                print(step)
            print("--- End Model Context Trace ---")

            if rac_corrector.rac_enabled:
                if any("Corrections Applied:" in step for step in mcp_response["trace"]):
                    rac_stats['corrected_queries'] += 1
                if any("Uncertain Claims Flagged:" in step for step in mcp_response["trace"]):
                    rac_stats['uncertain_claims_flagged'] += 1

            context_memory[user_question] = mcp_response["final_answer"]
        except Exception as e:
            logger.error(f"Error during interaction loop: {e}", exc_info=True)
            print("An unhandled error occurred while processing your request.")
            
        print("="*50 + "\n")
    logger.info("Enhanced Hybrid Chatbot with Model Context Protocol Session Completed.")

if __name__ == "__main__":
    main()
