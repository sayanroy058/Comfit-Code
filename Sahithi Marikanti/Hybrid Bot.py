
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
import hashlib # For caching claim hashes
import traceback # Import traceback for detailed error logging
import time # I am adding this import to check time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global Exception Hook for Tracebacks ---
def my_excepthook(type, value, traceback):
    import traceback as tb
    tb.print_exception(type, value, traceback)
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
from pydantic import BaseModel, Field
from llama_index.core import PromptTemplate # Import PromptTemplate

# --- MCP Schema Definitions (My Code) ---
class MCPRequest(BaseModel):
    """Represents a request formatted according to the Modular Cognitive Protocol."""
    query: str = Field(description="The user's query or instruction.")
    context_memory: Dict[str, Any] = Field(default_factory=dict, description="Persistent contextual information or long-term memory for the session.")
    tool_outputs: List[Dict] = Field(default_factory=list, description="Outputs from previous tool calls in the current step/turn.")
    scratchpad: str = Field(default="", description="Mutable workspace for intermediate thoughts, short-term memory, or agent trace.")

class MCPResponse(BaseModel):
    """Represents a response formatted according to the Modular Cognitive Protocol."""
    final_answer: str = Field(description="The consolidated, final answer to the user's query.")
    trace: List[str] = Field(default_factory=list, description="A chronological log of intermediate agent steps, thoughts, tool uses, and corrections.")
    confidence_score: float = Field(default=0.0, description="Overall confidence score for the final answer (0.0 to 1.0).")
    # I could add more fields like:
    # corrected_claims_info: List[Dict] = Field(default_factory=list, description="Details of claims that were corrected.")
    # uncertain_claims_info: List[Dict] = Field(default_factory=list, description="Details of claims with low confidence.")

# --- LLM Setup (My Code) ---
OPTIMAL_LLM_MODEL_NAME = "llama3"
OPTIMAL_LLM = Ollama(model=OPTIMAL_LLM_MODEL_NAME, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = OPTIMAL_LLM

# --- RAC (Retrieval-Augmented Correction) Implementation (My Code) ---

class FactualClaimExtractor:
    """ extracting atomic factual claims from LLM responses."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self, text: str) -> List[str]:
        """
        I extract atomic factual claims from the given text.
        I use an LLM to identify discrete, verifiable facts.
        """
        logger.info(" extracting factual claims from the response...")
        
        extraction_prompt = f"""
        Task: Extract atomic factual claims from the following text. 
        An atomic factual claim is a single, verifiable statement that can be true or false.
        
        Rules:
        1. Each claim should be independent and verifiable
        2. Break down complex sentences into simple facts
        3. Include numerical facts, dates, names, and specific details
        4. Ignore opinions, subjective statements, or procedural instructions.
        5. **CRITICAL**: Do NOT extract claims that describe the chatbot's internal process, tools used, or lack of information about a query (e.g., "There is no mention of X", "I used tool Y", "The tools mentioned are..."). Focus ONLY on claims about the subject matter.
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
                    if claim and len(claim) > 10:  # I filter out very short claims
                        claims.append(claim)
            
            logger.info(f"I extracted {len(claims)} factual claims")
            return claims
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

class FactVerifier:
    """ verifying fact claims against retrieved evidence."""
    
    def __init__(self, llm, local_query_engine, Google_Search_tool):
        self.llm = llm
        self.local_query_engine = local_query_engine
        self.Google_Search_tool = Google_Search_tool
        self.verification_cache = {} # I use this cache for storing verification results
        self.reversal_min_confidence = 0.95 # This is my new: High confidence required for a reversal correction

    def _get_claim_hash(self, claim: str) -> str:
        """I generate a unique hash for a claim to use as a cache key."""
        return hashlib.md5(claim.lower().encode('utf-8')).hexdigest()

    def verify_claim(self, claim: str, use_local: bool = True, use_web: bool = True) -> Dict[str, Any]:
        """
        I verify a single factual claim against available evidence sources.
        I use an in-session cache to avoid re-verifying the same claim.
        
        Returns:
            Dict containing verification results, evidence, and confidence score
        """
        claim_hash = self._get_claim_hash(claim)
        if claim_hash in self.verification_cache:
            logger.info(f"Cache hit for claim: {claim[:50]}... Returning cached result.")
            return self.verification_cache[claim_hash]

        logger.info(f" verifying claim: {claim[:50]}...")
        
        evidence = []
        
        # I try local knowledge base first with enhanced queries
        if use_local:
            try:
                logger.info(f"--- Local Knowledge:  checking the claim against PDF content ---")
                
                # I try multiple query strategies for better local retrieval
                queries_to_try = [
                    claim,  # Direct claim
                    f"What information is available about: {claim}",  # Question format
                    f"Find details related to: {self._extract_search_terms(claim)}",  # Key terms
                    f"Does the document mention: {self._extract_search_terms(claim)}"  # Mention check
                ]
                
                local_evidence_found = False
                for query in queries_to_try:
                    try:
                        local_result = self.local_query_engine.query(query)
                        local_content = str(local_result).strip()
                        
                        # I check if I got meaningful content (not just "I don't know" type responses)
                        if (local_content and 
                                len(local_content) > 20 and 
                                not any(phrase in local_content.lower() for phrase in [
                                    "i don't know", "no information", "not mentioned", 
                                    "cannot find", "not available", "no details"
                                ])):
                            evidence.append({
                                'source': 'local_knowledge',
                                'content': local_content,
                                'confidence': 0.9,  # I assign higher confidence for local data
                                'query_used': query
                            })
                            local_evidence_found = True
                            logger.info(f"✅ Local evidence found using query: {query[:50]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Local query failed for '{query[:30]}...': {e}")
                        continue
                
                if not local_evidence_found:
                    logger.info("⚠️ No relevant local evidence found in PDF")
                    
            except Exception as e:
                logger.warning(f"Local verification failed: {e}")
        
        # I try web search for additional evidence (only if local evidence is insufficient or as a fallback)
        if use_web and not local_evidence_found: # I only use web if local evidence wasn't conclusive
            try:
                logger.info(f"--- Web Search:  searching for additional evidence ---")
                # I extract key terms from the claim for better search
                search_query = self._extract_search_terms(claim)
                web_result = self.Google_Search_tool.search(search_query)
                if web_result and "No relevant search results" not in web_result:
                    evidence.append({
                        'source': 'web_search',
                        'content': web_result,
                        'confidence': 0.7,  # I assign lower confidence for web data
                        'query_used': search_query
                    })
                    logger.info(f"✅ Web evidence found for query: {search_query}")
            except Exception as e:
                logger.warning(f"Web verification failed: {e}")
        
        # I log evidence sources found
        evidence_sources = [e['source'] for e in evidence]
        logger.info(f"Evidence sources I used: {evidence_sources}")
        
        # I analyze evidence and determine if the claim is supported
        verification_result = self._analyze_evidence(claim, evidence)
        
        result_to_cache = {
            'claim': claim,
            'is_supported': verification_result['is_supported'],
            'confidence': verification_result['confidence'],
            'evidence': evidence,
            'correction_suggestion': verification_result.get('correction', None),
            'warning': verification_result.get('warning', None) # I add warning for low confidence
        }
        self.verification_cache[claim_hash] = result_to_cache
        return result_to_cache
    
    def _extract_search_terms(self, claim: str) -> str:
        """I extract key search terms from a claim."""
        # I remove common stop words and focus on key terms
        words = claim.split()
        # My simple heuristic: I keep words that are likely to be important
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        return ' '.join(key_words[:5])  # I limit to 5 key terms
    
    def _analyze_evidence(self, claim: str, evidence: List[Dict]) -> Dict[str, Any]:
        """
        I analyze evidence to determine if the claim is supported.
        I prioritize direct definitions from local documents.
        I include a stricter reversal sanity check.
        """
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
            evidence_text += "=== LOCAL DOCUMENT EVIDENCE (High Priority - Authoritative for specific terms) ===\n"
            for e in local_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        
        if web_evidence:
            evidence_text += "=== WEB SEARCH EVIDENCE (Lower Priority - General Knowledge) ===\n"
            for e in web_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        
        analysis_prompt = f"""
        Task: Analyze whether the following CLAIM is supported by the provided EVIDENCE.
        
        CLAIM: {claim}
        
        EVIDENCE:
        {evidence_text}
        
        Instructions:
        1. **ABSOLUTE PRIORITY**: If the LOCAL DOCUMENT EVIDENCE provides a **direct, clear, and unambiguous definition or specific factual information** for the CLAIM, especially if the claim involves an acronym or specialized term, consider that the most authoritative source. In such cases, if the local document directly supports the claim, the verdict should be SUPPORTED with high confidence, even if general web search results are ambiguous or different regarding the term's common usage.
        2. If local evidence supports/contradicts the claim, give it higher weight.
        3. Determine if the claim is SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE based on the evidence.
        4. Provide a confidence score (0.0 to 1.0).
        5. If contradicted, suggest a correction based on the evidence.
        6. Be precise and factual in your analysis.
        7. **Formal Correctness Note / Reversal Sanity Check**: If the evidence suggests a correction that *reverses the core meaning or polarity* of the original claim (e.g., from positive to negative, or vice versa), such a reversal requires exceptionally strong and unambiguous evidence. If the evidence for such a reversal is not overwhelming (e.g., confidence is below {self.reversal_min_confidence}), then you must default to INSUFFICIENT_EVIDENCE and provide no correction, as the original claim cannot be confidently disproven with a complete reversal.
        
        Respond in this format:
        VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE]
        CONFIDENCE: [0.0-1.0]
        CORRECTION: [If contradicted, provide corrected version, otherwise say "None"]
        REASONING: [Brief explanation including which evidence source was used]
        """
        
        try:
            response = str(self.llm.complete(analysis_prompt))
            
            # I parse the response
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
            
            # I boost confidence if local evidence was used and it was a direct support/contradiction
            if local_evidence and verdict in ["SUPPORTED", "CONTRADICTED"]:
                confidence = min(confidence + 0.2, 1.0)  # I boost by 0.2 for local evidence
            
            is_supported = verdict == "SUPPORTED"
            
            # --- Reversal Sanity Check Logic (My Enhanced Logic) ---
            # This is a placeholder. A more robust check involves semantic analysis.
            # Here, I'll rely heavily on the LLM's reasoning given the new prompt instruction.
            # If the LLM produces a correction, and it seems like a reversal,
            # I check if its *stated* confidence meets my high threshold.
            
            # My simple keyword-based detection for potential reversals (can be expanded)
            negative_markers = ["not", "no", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "never", "false", "untrue", "without"]
            
            original_has_neg = any(neg_m in claim.lower() for neg_m in negative_markers)
            correction_has_neg = any(neg_m in (correction or "").lower() for neg_m in negative_markers)

            if correction and ((original_has_neg and not correction_has_neg) or (not original_has_neg and correction_has_neg)):
                # This suggests a polarity reversal
                if confidence < self.reversal_min_confidence:
                    logger.warning(f"⚠️ Potential strong reversal detected. Original: '{claim}' -> Corrected: '{correction}'. Confidence ({confidence:.2f}) below strict reversal threshold ({self.reversal_min_confidence:.2f}). Suppressing correction.")
                    correction = None # I suppress the correction
                    verdict = "INSUFFICIENT_EVIDENCE" # I downgrade the verdict
                    confidence = 0.5 # I reset confidence to reflect uncertainty
                else:
                    logger.info(f"✅ Reversal correction accepted with high confidence ({confidence:.2f}). Original: '{claim}' -> Corrected: '{correction}'.")


            warning_message = None
            # I handle uncertain claims: claims that are not supported and have a confidence below a certain threshold
            # I'll use a `uncertainty_threshold` for this.
            if not is_supported and confidence <= 0.6: # Example threshold for warning
                warning_message = f"Low confidence in claim: {claim}. Verdict: {verdict}. Confidence: {confidence:.2f}. Consider further verification."
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
    """ the main RAC (Retrieval-Augmented Correction) system."""
    
    def __init__(self, llm, local_query_engine, Google_Search_tool):
        self.llm = llm
        self.claim_extractor = FactualClaimExtractor(llm)
        self.fact_verifier = FactVerifier(llm, local_query_engine, Google_Search_tool)
        self.correction_threshold = 0.5  # My lowered threshold for more corrections
        self.uncertainty_threshold = 0.6 # My new: Threshold below which a claim is considered uncertain
        self.local_priority = True  # I prioritize local evidence
        self.testing_mode = False # My new: Flag for testing mode
        
        # My new: Toggle for verification mode
        self.verification_mode = "hybrid" # Options: "hybrid", "local_only", "web_only"
        
    def correct_response(self, original_response: str, apply_corrections: bool = True) -> Dict[str, Any]:
        """
        I apply RAC correction to an LLM response.
        
        Args:
            original_response: The original LLM response to correct
            apply_corrections: Whether to apply corrections or just analyze (for testing mode)
            
        Returns:
            Dict containing corrected response and analysis details
        """
        logger.info("starting the RAC correction process...")
        
        # Step 1: I extract factual claims
        start_claim_extraction = time.perf_counter() #  timing this step
        claims = self.claim_extractor.extract_claims(original_response)
        end_claim_extraction = time.perf_counter() #  timing this step
        logger.info(f"Timing - Claim Extraction: {end_claim_extraction - start_claim_extraction:.4f} seconds")
        
        if not claims:
            logger.info("No factual claims extracted,  returning the original response")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 0.0 # Added by me for CC
            }
        
        # Step 2: I verify each claim
        verification_results = []
        corrections_needed = []
        uncertain_claims = []
        
        # I determine which sources to use based on the current mode
        use_local_source = self.verification_mode in ["hybrid", "local"]
        use_web_source = self.verification_mode in ["hybrid", "web"]

        if not use_local_source and not use_web_source:
            logger.warning("No verification sources enabled.  skipping claim verification.")
            for claim in claims: # I populate with default low confidence if no sources are active
                verification_results.append({
                    'claim': claim,
                    'is_supported': False,
                    'confidence': 0.0,
                    'correction_suggestion': None,
                    'warning': "No verification sources enabled."
                })
        else:
            start_verification = time.perf_counter() # timing this step
            for i, claim in enumerate(claims, 1):
                logger.info(f" processing claim {i}/{len(claims)}: {claim[:50]}...")
                start_single_verify = time.perf_counter() #  timing each single verification
                result = self.fact_verifier.verify_claim(claim, use_local=use_local_source, use_web=use_web_source)
                end_single_verify = time.perf_counter() #  timing each single verification
                logger.info(f"Timing - Single Claim Verification ({i}): {end_single_verify - start_single_verify:.4f} seconds")
                
                verification_results.append(result)
                
                # I log the verification result
                evidence_sources = [e['source'] for e in result['evidence']]
                logger.info(f"Claim {i} verification: {result['is_supported']}, "
                                f"confidence: {result['confidence']:.2f}, "
                                f"sources: {evidence_sources}")
                
                if not result['is_supported'] and result['confidence'] > self.correction_threshold:
                    if result['correction_suggestion']:
                        corrections_needed.append({
                            'original_claim': claim,
                            'correction': result['correction_suggestion'],
                            'confidence': result['confidence'],
                            'evidence_sources': evidence_sources
                        })
                        logger.info(f"✏️ Correction needed for claim {i}")
                    else:
                        logger.info(f"⚠️ Claim {i} not supported but no correction available")
                
                # I check for uncertain claims
                if result['confidence'] < self.uncertainty_threshold:
                    uncertain_claims.append({
                        'claim': claim,
                        'confidence': result['confidence'],
                        'verdict': "SUPPORTED" if result['is_supported'] else "CONTRADICTED" if result['correction_suggestion'] else "INSUFFICIENT_EVIDENCE",
                        'warning': result.get('warning', 'Low confidence.')
                    })
                    logger.warning(f"❗️ Claim {i} is uncertain: {claim[:50]}... Confidence: {result['confidence']:.2f}")
            end_verification = time.perf_counter() #  timing this step
            logger.info(f"Timing - All Claims Verification: {end_verification - start_verification:.4f} seconds")

        # Step 3: I apply corrections if requested and not in testing mode
        corrected_response = original_response
        if apply_corrections and not self.testing_mode and corrections_needed:
            start_apply_corrections = time.perf_counter() # timing this step
            corrected_response = self._apply_corrections(original_response, corrections_needed)
            end_apply_corrections = time.perf_counter() #  timing this step
            logger.info(f"Timing - Applying Corrections: {end_apply_corrections - start_apply_corrections:.4f} seconds")
            
        logger.info(f"RAC correction completed. I analyzed {len(claims)} claims, made {len(corrections_needed)} corrections")
        
        # I calculate average confidence for Confidence Cascade
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
            'average_confidence': average_confidence # Added by me for CC
        }
    
    def _apply_corrections(self, original_response: str, corrections: List[Dict]) -> str:
        """I apply corrections to the original response."""
        correction_prompt = f"""
        Task: Apply the following corrections to the original response while maintaining its structure and flow.
        
        ORIGINAL RESPONSE:
        {original_response}
        
        CORRECTIONS TO APPLY:
        {chr(10).join([f"- Replace/correct: '{c['original_claim']}' → '{c['correction']}'" for c in corrections])}
        
        Instructions:
        1. Integrate the corrections naturally into the response
        2. Maintain the original tone and structure
        3. Ensure the corrected response flows well
        4. Don't add unnecessary information
        5. Keep the response length similar to the original
        6. **Formal Correctness Note**: Do NOT include any meta-information about the correction process itself (e.g., "This response has been corrected", "Based on verification"). Just provide the corrected information.
        
        Provide the corrected response:
        """
        
        try:
            corrected = str(self.llm.complete(correction_prompt))
            return corrected.strip()
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return original_response

# --- Enhanced Google Search Tool (My Code) ---
def validate_google_api_keys_from_env():
    """I validate the presence of Google API keys in environment variables."""
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
    """ a tool to perform Google Custom Searches."""
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> str:
        """
        I perform a Google Custom Search and return formatted results.
        I use this tool for questions that require up-to-date or broad web information.
        """
        logger.info(f"--- Google Search:  searching for '{query}' ---")
        start_web_api = time.perf_counter() #  timing the web API call
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
        finally:
            end_web_api = time.perf_counter() # timing the web API call
            logger.info(f"Timing - Google Web Search API Call: {end_web_api - start_web_api:.4f} seconds")

# --- PDF Processing Functions (My Code - unchanged logic) ---
def clean_text(text):
    """I clean text by removing hyphens, newlines, page numbers, and excessive whitespace."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[.!?]\n', '. ', text)
    text = re.sub(r'[,;]\n', ', ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def curate_pdf_to_text(pdf_path, output_dir):
    """
    I extract text from a PDF, clean it, and save it to a .txt file.
    """
    txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
    output_filepath = os.path.join(output_dir, txt_filename)
    logger.info(f" processing PDF: {os.path.basename(pdf_path)}...")
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
            logger.warning(f"Extracted and cleaned text from '{pdf_path}' is empty. I am skipping.")
            return None
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(final_curated_text)
        logger.info(f"I successfully curated and saved text to: {output_filepath}")
        return output_filepath
    except PyPDF2.errors.PdfReadError:
        logger.critical(f"FATAL ERROR: I could not read PDF file '{pdf_path}'. It might be encrypted or corrupted. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL ERROR: An unexpected error occurred while processing '{pdf_path}': {e}. Exiting.")
        sys.exit(1)

def load_single_document_for_indexing(file_path: str):
    """
    I load a single text document using SimpleDirectoryReader for LlamaIndex indexing.
    """
    if not os.path.exists(file_path):
        logger.critical(f"FATAL ERROR: Processed text file '{file_path}' not found. Exiting.")
        sys.exit(1)
    logger.info(f"loading document for indexing: {os.path.basename(file_path)}")
    reader = SimpleDirectoryReader(input_files=[file_path], required_exts=[".txt"])
    documents = reader.load_data()
    if not documents:
        logger.critical(f"FATAL ERROR: No content could be loaded from '{file_path}' for indexing. Exiting.")
        sys.exit(1)
    filename = os.path.basename(file_path)
    dummy_category = "BookContent"
    for doc in documents:
        doc.metadata['category'] = dummy_category
        doc.metadata['filename'] = filename
    logger.info(f"I loaded {len(documents)} document segments for indexing from '{filename}'.")
    return documents

# --- Centralized MCP Processing Function (My Code) ---
def process_mcp_query(
    mcp_request: MCPRequest, 
    agent_instance: ReActAgent, 
    rac_corrector_instance: 'RACCorrector', 
    testing_mode: bool,
    suppress_threshold: float,
    flag_threshold: float
) -> MCPResponse:
    """
    I process an MCPRequest through the agent and RAC pipeline,
    returning an MCPResponse.
    """
    user_query = mcp_request.query
    response_trace = [f"MCPRequest received: Query='{user_query}'"]

    start_total_process_mcp = time.perf_counter() #  starting the overall timer for this function

    try:
        # Step A: I pre-process the user question to encourage local search for definitions
        start_preprocess = time.perf_counter() # timing preprocessing
        processed_question = user_query
        pdf_specific_keywords = ["speed process", "anthropometry", "product fit", "sizing", "my document", "this document"]
        is_definitional_phrase = any(term in user_query.lower() for term in ["what is", "what does", "define", "stands for", "meaning of"])
        is_pdf_specific_query = any(pdf_term in user_query.lower() for pdf_term in pdf_specific_keywords)

        if is_definitional_phrase and is_pdf_specific_query:
            processed_question = f"According to the provided document, {user_query}"
            logger.info(f"I adjusted the question for local priority: '{processed_question}'")
        else:
            logger.info(f"Question not adjusted for local priority (general query): '{user_query}'")
        response_trace.append(f"Pre-processed query for agent: '{processed_question}'")
        end_preprocess = time.perf_counter() #  timing preprocessing
        response_trace.append(f"Timing - Preprocessing: {end_preprocess - start_preprocess:.4f} seconds")


        # I get the initial response from the agent
        start_agent_response = time.perf_counter() #  timing agent response generation
        agent_response_obj = agent_instance.chat(processed_question)
        original_response_text = agent_response_obj.response
        end_agent_response = time.perf_counter() #  timing agent response generation
        response_trace.append(f"Agent raw response: '{original_response_text}'")
        response_trace.append(f"Timing - Agent Raw Response Generation: {end_agent_response - start_agent_response:.4f} seconds")

        # I apply RAC correction if enabled
        if rac_corrector_instance.rac_enabled: # I check RAC status from corrector instance if I manage it there
            start_rac = time.perf_counter() #  timing the overall RAC process
            response_trace.append(" applying RAC (Retrieval-Augmented Correction)...")
            rac_result = rac_corrector_instance.correct_response(original_response_text, apply_corrections=not testing_mode)
            end_rac = time.perf_counter() # timing the overall RAC process
            response_trace.append(f"Timing - RAC Process: {end_rac - start_rac:.4f} seconds")
            
            # I add RAC specific trace info
            response_trace.append(f"RAC Analysis: {rac_result['claims_analyzed']} claims checked.")
            if rac_result['corrections_made'] > 0:
                response_trace.append(f"  Corrections Applied: {rac_result['corrections_made']}")
                for corr in rac_result['corrections_applied']:
                    response_trace.append(f"    - Original: '{corr['original_claim'][:70]}...' -> Corrected: '{corr['correction'][:70]}...'")
            if rac_result['uncertain_claims']:
                response_trace.append(f"  Uncertain Claims Flagged: {len(rac_result['uncertain_claims'])}")
                for uc in rac_result['uncertain_claims']:
                    response_trace.append(f"    - Claim: '{uc['claim'][:70]}...' (Conf: {uc['confidence']:.2f})")
            
            # Confidence Cascade Decision Logic (My Code)
            average_conf = rac_result['average_confidence']
            final_answer_content = ""

            if average_conf < suppress_threshold:
                final_answer_content = f"❌ Response suppressed due to very low confidence ({average_conf:.2f}). The system lacks sufficient confidence to provide a reliable answer for this query."
                response_trace.append(f"Confidence Cascade: Response Suppressed (Avg Confidence: {average_conf:.2f})")
            elif average_conf < flag_threshold:
                final_answer_content = rac_result['corrected_response'] if rac_result['corrections_made'] > 0 else original_response_text
                final_answer_content = f"⚠️ Low confidence in response ({average_conf:.2f}). Please use with caution.\n\n" + final_answer_content
                response_trace.append(f"Confidence Cascade: Response Flagged (Avg Confidence: {average_conf:.2f})")
            else:
                final_answer_content = rac_result['corrected_response'] if rac_result['corrections_made'] > 0 else original_response_text
                response_trace.append(f"Confidence Cascade: Response Accepted (Avg Confidence: {average_conf:.2f})")

            end_total_process_mcp = time.perf_counter() #  ending the overall timer for this function
            response_trace.append(f"Timing - Total process_mcp_query duration (incl. RAC): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")

            return MCPResponse(
                final_answer=final_answer_content,
                trace=response_trace,
                confidence_score=average_conf
            )
        else: # RAC is disabled
            response_trace.append("RAC Disabled,  returning the agent's raw response.")
            end_total_process_mcp = time.perf_counter() # ending the overall timer for this function
            response_trace.append(f"Timing - Total process_mcp_query duration (RAC Disabled): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")
            return MCPResponse(
                final_answer=original_response_text,
                trace=response_trace,
                confidence_score=1.0 # I assume full confidence if RAC is disabled, or use agent's internal confidence
            )
            
    except Exception as e:
        logger.error(f"Error in process_mcp_query: {e}", exc_info=True) # I log the full traceback
        error_message = "An unexpected error occurred while processing your request."
        response_trace.append(f"ERROR: {e}")
        response_trace.append(traceback.format_exc()) # I add the full traceback to the trace for debugging
        end_total_process_mcp = time.perf_counter() #  ending the overall timer even on error
        response_trace.append(f"Timing - Total process_mcp_query duration (Error): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")
        return MCPResponse(
            final_answer=error_message,
            trace=response_trace,
            confidence_score=0.0 # I set lowest confidence on error
        )


def main():
    """My main function to run the Enhanced Hybrid Chatbot with RAC."""
    logger.info(" starting the Enhanced Hybrid Chatbot with RAC (Retrieval-Augmented Correction)...")

    if len(sys.argv) < 2:
        logger.critical("FATAL ERROR: Please provide the path to your PDF book as a command-line argument.")
        logger.critical("Example: python mcp_rac.py \"./ebooks/Product Fit and Sizing_25_06_03_12_44_21.pdf\"")
        sys.exit(1)
    pdf_file_path = sys.argv[1]

    # My new: I parse for testing mode flag
    testing_mode_enabled = "--dry-run" in sys.argv
    if testing_mode_enabled:
        logger.info("RAC Testing Mode (--dry-run) enabled: Corrections will be analyzed but NOT applied.")
        sys.argv.remove("--dry-run") # I remove the flag so it doesn't interfere with other parsing

    CURATED_DATA_SINGLE_BOOK_DIR = 'curated_data_single_book'
    os.makedirs(CURATED_DATA_SINGLE_BOOK_DIR, exist_ok=True)

    # I process the PDF to a text file
    processed_text_file_path = curate_pdf_to_text(pdf_file_path, CURATED_DATA_SINGLE_BOOK_DIR)
    if not processed_text_file_path:
        logger.critical(f"FATAL ERROR: I could not process PDF '{pdf_file_path}'. Exiting.")
        sys.exit(1)

    # --- Step 1: I Load and Index Local Document ---
    documents = load_single_document_for_indexing(processed_text_file_path)
    logger.info(" creating VectorStoreIndex for local PDF data...")
    try:
        local_index = VectorStoreIndex.from_documents(
            documents,
            llm=OPTIMAL_LLM,
            embed_model=Settings.embed_model,
        )
        local_query_engine = local_index.as_query_engine(llm=OPTIMAL_LLM)
        logger.info("Local PDF data indexed successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: I could not create VectorStoreIndex for local data: {e}. Exiting.")
        sys.exit(1)

    # --- Step 2: I Initialize Google Search Tool ---
    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. I cannot perform web searches. Exiting.")
        sys.exit(1)
    
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5
    )

    # --- Step 3: I Initialize RAC System ---
    rac_corrector = RACCorrector(
        llm=OPTIMAL_LLM,
        local_query_engine=local_query_engine,
        Google_Search_tool=Google_Search_instance
    )
    rac_corrector.testing_mode = testing_mode_enabled # I set testing mode

    # --- Step 4: I Define Tools for the Agent ---
    class LocalBookQAToolInput(BaseModel):
        query: str = Field(description="The question to ask about the PDF book content.")

    def local_book_qa_function(query: str) -> str:
        """
        I use this for questions specifically about the content of the provided PDF book.
        """
        logger.info(f"--- Local RAG:  querying for '{query}' ---")
        start_local_rag_query = time.perf_counter() #  timing local RAG query
        response = local_query_engine.query(query)
        end_local_rag_query = time.perf_counter() #  timing local RAG query
        logger.info(f"Timing - Local RAG Query: {end_local_rag_query - start_local_rag_query:.4f} seconds")
        return str(response)

    local_rag_tool = FunctionTool.from_defaults(
        fn=local_book_qa_function,
        name="local_book_qa",
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "Use this tool when the user's question relates to information likely found "
            "within the PDF document."
        ),
        fn_schema=LocalBookQAToolInput,
    )

    Google_Search_tool_for_agent = FunctionTool.from_defaults(
        fn=Google_Search_instance.search,
        name="google_web_search",
        description=(
            "Useful for answering general knowledge questions, current events, or anything "
            "that requires searching the internet. Always consider using this for questions "
            "that are not likely to be in the provided PDF book, or to verify facts."
        ),
    )

    tools = [local_rag_tool, Google_Search_tool_for_agent]

    # --- Step 5: Initialize the ReAct Agent ---
    logger.info(f"initializing ReAct Agent with LLM: {OPTIMAL_LLM_MODEL_NAME}...")
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=OPTIMAL_LLM,
        verbose=False, #  set to True for debugging to see agent's thoughts and tool calls
        max_iterations=30 # I keep at 10 for now, as the RAC is handling post-processing
    )

    # --- Step 5.1:  Customize the Agent's System Prompt for Formal Correctness ---
    #  craft a new system prompt that explicitly forbids mentioning tools/internal process
    strict_agent_system_template = PromptTemplate(
        """You are a helpful and factual assistant that provides answers based solely on the information you find.
        You have access to various tools to help you gather information.
        
        **CRITICAL INSTRUCTION**: When providing your final answer, do NOT mention the names of the tools you used (e.g., 'local_book_qa', 'google_web_search'), your internal thoughts, observations, or action steps. Your response should be a direct, clear, and concise answer to the user's question, formulated as if you already know the information.
        
        If you cannot find information about a specific query in the provided sources, state clearly and politely that the information is not available in your knowledge base (e.g., "Information about [topic] was not found in the provided documents/sources.").
        Do not speculate or provide information that is not directly supported by the data.
        
        {tool_desc}

        ## Output Format
        To answer the question, please use the following format.

        Thought: I need to use a tool to help me answer the question.
        Action: tool_name
        Action Input: {{ "param": "value" }}
        Observation: Tool output goes here.
        ... (this Thought/Action/Observation can repeat multiple times)
        Thought: I can answer without using any more tools.
        Answer: [Your final answer here, strictly adhering to the CRITICAL INSTRUCTION above]
        """
    )

    # set the custom system prompt for the agent
    try:
        agent_prompts = agent.get_prompts()
        agent_prompts["agent_worker:system_prompt"] = strict_agent_system_template
        agent.set_prompts(agent_prompts)
        logger.info("Agent's system prompt updated for stricter output.")
    except Exception as e:
        logger.error(f"Failed to update agent's system prompt: {e}. Tool mentions might persist. Ensure LlamaIndex version is compatible or adapt prompt setting method.")

    logger.info("I initialized the Enhanced ReAct Agent with RAC.")

    logger.info("\n--- Enhanced Hybrid Chatbot with RAC READY ---")
    logger.info(f"Agent uses LLM: {OPTIMAL_LLM_MODEL_NAME}")
    logger.info(f"Tools available: local_book_qa, google_web_search")
    logger.info(f"RAC (Retrieval-Augmented Correction) enabled for fact-checking ({'Testing Mode' if testing_mode_enabled else 'Active'})")
    logger.info(f"Type your questions. Type 'exit' to quit.")
    logger.info(f"Commands: 'toggle_rac' to enable/disable RAC, 'rac_stats' for statistics, 'set_mode [hybrid|local|web]' to change verification mode")

    # --- Step 6: Interactive Query Loop with RAC (My Code) ---
    # rac_enabled is now managed within rac_corrector_instance for MCP clarity
    rac_corrector.rac_enabled = True # I initialize RAC as enabled by default in the corrector instance
    rac_stats = {
        'total_queries': 0, 
        'corrected_queries': 0, 
        'total_corrections': 0, 
        'uncertain_claims_flagged': 0,
        'responses_suppressed': 0, # My new stat
        'responses_flagged_low_confidence': 0 # My new stat
    }

    # Confidence Cascade Thresholds (My Code)
    SUPPRESS_THRESHOLD = 0.4 # Below this, response is suppressed
    FLAG_THRESHOLD = 0.6     # Below this, response is flagged for low confidence

    while True:
        user_question = input("\nEnter your question (or 'exit'): ").strip()
        
        if user_question.lower() == 'exit':
            logger.info(" exiting the Enhanced Hybrid Chatbot. Goodbye!")
            break
        elif user_question.lower() == 'toggle_rac':
            rac_corrector.rac_enabled = not rac_corrector.rac_enabled
            print(f"RAC is now {'ENABLED' if rac_corrector.rac_enabled else 'DISABLED'}")
            continue
        elif user_question.lower() == 'rac_stats':
            print(f"\n--- RAC Statistics ---")
            print(f"Total queries processed: {rac_stats['total_queries']}")
            print(f"Queries with corrections: {rac_stats['corrected_queries']}")
            print(f"Total corrections made: {rac_stats['total_corrections']}")
            print(f"Total uncertain claims flagged: {rac_stats['uncertain_claims_flagged']}")
            print(f"Responses suppressed (very low confidence): {rac_stats['responses_suppressed']}")
            print(f"Responses flagged (low confidence, usable): {rac_stats['responses_flagged_low_confidence']}")
            print(f"Correction rate: {rac_stats['corrected_queries']/max(rac_stats['total_queries'], 1)*100:.1f}%")
            continue
        elif user_question.lower().startswith('set_mode'):
            parts = user_question.lower().split()
            if len(parts) == 2 and parts[0] == 'set_mode':
                mode = parts[1]
                if mode in ["hybrid", "local", "web"]:
                    rac_corrector.verification_mode = mode
                    print(f"Verification mode set to: {mode.upper()}.")
                    if mode == "local":
                        print("RAC will now only use local PDF for verification.")
                    elif mode == "web":
                        print("RAC will now only use web search for verification.")
                    else:
                        print("RAC will use both local PDF and web search for verification (hybrid).")
                else:
                    print("Invalid mode. Use 'set_mode [hybrid|local|web]'.")
            else:
                print("Invalid command format. Use 'set_mode [hybrid|local|web]'.")
            continue

        print("\n" + "="*50)
        print("--- Agent's Response (via MCP Interface) ---")
        print("="*50)
        
        try:
            #  prepare the MCP Request (for this simple loop, context/tools/scratchpad are minimal)
            mcp_request = MCPRequest(query=user_question)

            #  am adding these lines to check the total time for process_mcp_query
            start_time = time.time() 

            #  call the centralized MCP processing function
            mcp_response = process_mcp_query(
                mcp_request=mcp_request,
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=testing_mode_enabled,
                suppress_threshold=SUPPRESS_THRESHOLD,
                flag_threshold=FLAG_THRESHOLD
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️ Total processing time for this query: {elapsed_time:.2f} seconds") # Print total elapsed time for the query
            
            #  update statistics based on MCPResponse
            rac_stats['total_queries'] += 1 
            
            # display the final answer to the user
            print(mcp_response.final_answer)

            # display the trace for debugging/formal correctness
            print("\n--- MCP Trace ---")
            for step in mcp_response.trace:
                print(step)
            print("--- End MCP Trace ---")

            #  update specialized stats from the MCP response
            if mcp_response.confidence_score < SUPPRESS_THRESHOLD and mcp_response.final_answer.startswith("❌"):
                rac_stats['responses_suppressed'] += 1
            elif mcp_response.confidence_score < FLAG_THRESHOLD and mcp_response.final_answer.startswith("⚠️"):
                rac_stats['responses_flagged_low_confidence'] += 1

        except Exception as e:
            logger.error(f"Error during overall interaction loop: {e}", exc_info=True)
            print("An unhandled error occurred while processing your request.")
            
        print("="*50 + "\n")

    logger.info("Enhanced Hybrid Chatbot with RAC Session Completed.")

if __name__ == "__main__":
    main()
