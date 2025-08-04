import pandas as pd
import zipfile
import os
import json
import re
import uuid
import logging
from typing import BinaryIO, Dict, List, Optional, Union
from tempfile import NamedTemporaryFile

import google.generativeai as genai
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Local imports
import vector_db
from db_crud import get_follow_up_chats
from db import db_dependency

# Configuration
load_dotenv()
logger = logging.getLogger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genaimodel = genai.GenerativeModel(model_name='gemini-2.0-flash')

templates = Jinja2Templates(directory="templates")

# Constants
MAX_CONTEXT_LENGTH = 3000
DEFAULT_TOP_K = 10


def embed_and_store_csv(file: BinaryIO, filename: str) -> str:
    """
    Process CSV file and store embeddings in vector database.
    
    Args:
        file: Binary file object
        filename: Name of the file
        
    Returns:
        Success message string
    """
    try:
        # Read CSV with better error handling
        df = pd.read_csv(file, encoding='utf-8')
        
        if "clause_text" not in df.columns:
            raise ValueError("Missing required 'clause_text' column.")
        
        # Check for existing embeddings
        try:
            existing = vector_db.get(
                where={"source_file": filename},
                limit=1
            )
            
            if existing and existing.get('ids'):
                raise ValueError(f"Embeddings from '{filename}' already exist.")
        except Exception as e:
            # If it's not a "already exists" error, just log and continue
            if "already exist" not in str(e):
                logger.warning(f"Could not check existing embeddings: {e}")
        
        # Clean and prepare data
        df = df.dropna(subset=["clause_text"])
        df['clause_text'] = df['clause_text'].astype(str).str.strip()
        df = df[df['clause_text'].str.len() > 0]
        
        if df.empty:
            raise ValueError("No valid clause text found in CSV.")
        
        texts = df['clause_text'].tolist()
        types = df.get('clause_type', ['Unknown'] * len(texts)).fillna('Unknown').tolist()
        
        # Generate metadata
        metadatas = []
        for idx, clause_type in enumerate(types):
            metadatas.append({
                "type": str(clause_type),
                "clause_type": str(clause_type),
                "source_file": filename,
                "row_index": idx
            })
        
        # Store in vector database
        vector_db.add(
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=metadatas,
            documents=texts
        )
        
        logger.info(f"Successfully processed {len(texts)} clauses from {filename}")
        return f"✅ {len(texts)} clauses from '{filename}' added successfully."
        
    except Exception as e:
        logger.error(f"Error processing CSV {filename}: {str(e)}")
        raise ValueError(f"Failed to process CSV: {str(e)}")


def handle_uploaded_file(file: BinaryIO, filename: str) -> str:
    """Determine file type and process accordingly."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in [".csv", ".zip"]:
        raise ValueError("Only .csv and .zip files are allowed.")
    
    try:
        if ext == ".csv":
            return embed_and_store_csv(file, filename)
            
        elif ext == ".zip":
            results = []
            
            # Save the uploaded zip temporarily
            with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(file.read())
                tmp.flush()
                
                try:
                    with zipfile.ZipFile(tmp.name, "r") as zip_ref:
                        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]
                        
                        if not csv_files:
                            raise ValueError("No CSV files found in the zip.")
                        
                        for csv_file in csv_files:
                            try:
                                with zip_ref.open(csv_file) as f:
                                    result = embed_and_store_csv(f, os.path.basename(csv_file))
                                    results.append(result)
                            except Exception as e:
                                error_msg = f"❌ Failed to process {csv_file}: {str(e)}"
                                results.append(error_msg)
                                logger.warning(error_msg)
                    
                    return "\n".join(results)
                    
                finally:
                    os.remove(tmp.name)
                    
    except Exception as e:
        logger.error(f"Error handling file {filename}: {str(e)}")
        raise


def analyze_query_intent_and_chunks(user_query: str) -> Dict:
    """
    Analyze user query to determine intent and extract chunks.
    
    Args:
        user_query: User's input query
        
    Returns:
        Dictionary with intent and chunks
    """
    prompt = f"""
    You are an intelligent NLP assistant. Given a user query, do the following:

    1. Determine the user's intent from these options:
       - "compare": if the user is explicitly comparing two things
       - "define": if the user wants a definition or explanation
       - "retrieve": if the user wants to know applications, uses, benefits, etc.

    2. Extract key concepts based on intent:
       - If "compare" → return two separate concepts
       - If "define" or "retrieve" → return one main concept

    3. Return output ONLY in this JSON format:
    {{
        "intent": "compare|define|retrieve",
        "chunks": ["first", "second"] or ["single"],
        "confidence": 0.8
    }}

    User query: "{user_query}"
    """

    try:
        response = genaimodel.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 300
            }
        )
        
        raw_text = response.text.strip()
        
        # Extract JSON
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found")
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Result is not a dictionary")
            
        intent = result.get("intent", "retrieve")
        chunks = result.get("chunks", [user_query])
        
        # Ensure chunks is a list
        if not isinstance(chunks, list):
            chunks = [str(chunks)]
            
        # Clean empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk and str(chunk).strip()]
        if not chunks:
            chunks = [user_query]
        
        # Enforce single chunk for define/retrieve
        if intent in ("define", "retrieve") and len(chunks) > 1:
            chunks = [" ".join(chunks)]
        
        return {
            "intent": intent,
            "chunks": chunks,
            "confidence": result.get("confidence", 0.5)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing query intent: {str(e)}")
        return {
            "intent": "retrieve",
            "chunks": [user_query],
            "confidence": 0.0
        }


def search_top_k(result: Dict, k: int = DEFAULT_TOP_K) -> Dict:
    """
    Search for similar documents from vector database.
    
    Args:
        result: Dictionary containing intent and chunks
        k: Number of top results to return per chunk
        
    Returns:
        Dictionary with search results
    """
    # Input validation
    if not result or not isinstance(result, dict):
        logger.warning("Invalid result provided to search_top_k")
        return {"intent": "unknown", "results": [], "error": "Invalid input"}
    
    chunks = result.get("chunks", [])
    if not chunks:
        logger.warning("No chunks provided for search")
        return {"intent": result.get("intent", "unknown"), "results": [], "error": "No chunks"}
    
    intent = result.get("intent", "unknown")
    logger.info(f"Searching for {len(chunks)} chunks with intent '{intent}'")
    

    
    all_results = []
    
    for i, chunk in enumerate(chunks):
        if not chunk or not str(chunk).strip():
            logger.warning(f"Skipping empty chunk {i+1}")
            continue
            
        try:
            logger.info(f"Searching for chunk {i+1}: '{str(chunk)[:100]}...'")
            
            # Perform similarity search
            search_result = vector_db.similarity_search(str(chunk), n_results=k)
            
            # Handle the search result safely
            if not search_result or not isinstance(search_result, dict):
                logger.warning(f"Invalid search result for chunk {i+1}")
                continue
            
            # Extract results with safe indexing
            documents = search_result.get("documents", [])
            metadatas = search_result.get("metadatas", [])
            distances = search_result.get("distances", [])
            
            # Handle nested list structure (ChromaDB returns nested lists)
            if documents and isinstance(documents[0], list):
                documents = documents[0] if documents else []
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0] if metadatas else []
            if distances and isinstance(distances[0], list):
                distances = distances[0] if distances else []
            
            if not documents:
                logger.warning(f"No documents found for chunk {i+1}")
                continue
            
            # Process results
            chunk_results = []
            for j, doc in enumerate(documents):
                try:
                    if not doc or not str(doc).strip():
                        continue
                    
                    # Safe metadata extraction
                    meta = metadatas[j] if j < len(metadatas) else {}
                    score = distances[j] if j < len(distances) else 1.0
                    
                    # Ensure meta is a dictionary
                    if not isinstance(meta, dict):
                        meta = {}
                    
                    # Extract metadata safely
                    clause_type = meta.get("clause_type") or meta.get("type", "Unknown")
                    source_file = meta.get("source_file", "")
                    row_index = meta.get("row_index", -1)
                    
                    result_item = {
                        "score": float(score) if score is not None else 1.0,
                        "clause_text": str(doc).strip(),
                        "clause_type": str(clause_type),
                        "source_file": str(source_file),
                        "row_index": int(row_index) if isinstance(row_index, (int, float)) else -1,
                        "chunk_index": i + 1
                    }
                    
                    chunk_results.append(result_item)
                    
                except Exception as e:
                    logger.error(f"Error processing result {j} for chunk {i+1}: {e}")
                    continue
            
            logger.info(f"Found {len(chunk_results)} results for chunk {i+1}")
            
            # For single chunk or define/retrieve intents, return directly
            if intent in ("define", "retrieve") or len(chunks) == 1:
                return {
                    "intent": intent,
                    "chunks": chunks,
                    "results": chunk_results,
                    "total_results": len(chunk_results)
                }
            
            # For multi-chunk results (compare), organize by chunk
            if chunk_results:
                chunk_label = f"chunk_{i+1}"
                all_results.append({chunk_label: chunk_results})
            
        except Exception as e:
            logger.error(f"Error searching for chunk {i+1}: {e}")
            continue
    
    total_results = sum(len(list(chunk_data.values())[0]) for chunk_data in all_results)
    logger.info(f"Search completed: {total_results} total results across {len(all_results)} chunks")
    
    return {
        "intent": intent,
        "chunks": chunks,
        "results": all_results if all_results else [],
        "total_results": total_results
    }


def get_context_from_search_results(search_results: Dict, max_context_length: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Extract context string from search results for RAG.
    
    Args:
        search_results: Results from search_top_k function
        max_context_length: Maximum length of context string
        
    Returns:
        String containing relevant context
    """
    if not search_results or not search_results.get('results'):
        logger.warning("No search results provided for context extraction")
        return ""
    
    context_parts = []
    results = search_results['results']
    
    try:
        # Handle single chunk results (list of dicts)
        if isinstance(results, list) and results:
            # Check if it's a direct list of results or multi-chunk structure
            first_result = results[0]
            
            if isinstance(first_result, dict) and 'clause_text' in first_result:
                # Direct list of results
                for result in results[:5]:  # Top 5 results
                    if result.get('clause_text'):
                        context_parts.append(result['clause_text'])
            else:
                # Multi-chunk results
                for chunk_data in results:
                    if isinstance(chunk_data, dict):
                        for chunk_name, chunk_results in chunk_data.items():
                            if isinstance(chunk_results, list):
                                for result in chunk_results[:3]:  # Top 3 per chunk
                                    if isinstance(result, dict) and result.get('clause_text'):
                                        context_parts.append(result['clause_text'])
    
    except Exception as e:
        logger.error(f"Error extracting context: {e}")
        return ""
    
    # Join and truncate context
    full_context = "\n\n".join(context_parts)
    
    if len(full_context) > max_context_length:
        full_context = full_context[:max_context_length] + "..."
    
    logger.info(f"Generated context with {len(context_parts)} clauses, {len(full_context)} characters")
    return full_context


def format_rag_context(retrieval: Dict) -> str:
    """
    Format search results into readable context for RAG.
    
    Args:
        retrieval: Search results dictionary
        
    Returns:
        Formatted context string
    """
    if not retrieval or not retrieval.get("results"):
        return "No relevant clauses found."
    
    intent = retrieval.get("intent", "unknown")
    results = retrieval.get("results", [])
    context = ""
    
    try:
        # Handle single chunk results (define/retrieve)
        if isinstance(results, list) and results:
            first_result = results[0]
            
            if isinstance(first_result, dict) and "clause_text" in first_result:
                context += "### Retrieved Legal Clauses:\n"
                for i, result in enumerate(results[:10], 1):  # Limit to 10 results
                    if result.get('clause_text'):
                        context += f"{i}. {result['clause_text'].strip()}"
                        if result.get("clause_type"):
                            context += f" (Type: {result['clause_type']})"
                        if result.get("source_file"):
                            context += f" — from {result['source_file']}"
                        context += "\n"
                return context.strip()
        
        # Handle multi-chunk results (compare)
        for entry in results:
            if isinstance(entry, dict):
                for label, matches in entry.items():
                    context += f"\n### Results for {label.replace('_', ' ').title()}:\n"
                    if isinstance(matches, list):
                        for i, match in enumerate(matches[:5], 1):  # Top 5 per chunk
                            if isinstance(match, dict) and match.get('clause_text'):
                                context += f"{i}. {match['clause_text'].strip()}"
                                if match.get("clause_type"):
                                    context += f" (Type: {match['clause_type']})"
                                if match.get("source_file"):
                                    context += f" — from {match['source_file']}"
                                context += "\n"
    
    except Exception as e:
        logger.error(f"Error formatting context: {e}")
        return "Error formatting search results."
    
    return context.strip() if context.strip() else "No relevant clauses found."


def answer_with_gemini_rag(
    user_query: str, 
    context: str,
    follow_up_chats: Optional[str] = None,
    session_id: Optional[int] = None,
    db: Optional[db_dependency] = None
) -> str:
    """
    Generate a legal answer using Gemini with RAG.
    
    Args:
        user_query: The user's legal question
        context: Relevant legal clauses for grounding
        follow_up_chats: Previous chat history (optional)
        session_id: Current session ID (optional)
        db: Database dependency (optional)
        
    Returns:
        Legal answer with citations
    """
    # Get follow-up chats if session_id provided
    if session_id and db and not follow_up_chats:
        try:
            follow_up_chats = get_follow_up_chats(db, session_id)
            if follow_up_chats == "No follow-up chats found for this session.":
                follow_up_chats = None
        except Exception as e:
            logger.warning(f"Could not retrieve follow-up chats: {e}")
            follow_up_chats = None
    
    # Build prompt
    prompt = f"""
    LEGAL ANALYSIS REQUEST
    ----------------------
    You are a senior legal AI assistant specializing in contract law interpretation.
    
    RELEVANT LEGAL CONTEXT:
    {context if context else "No specific legal context provided."}
    
    CURRENT QUERY:
    {user_query}
    
    PREVIOUS CHAT HISTORY:
    {follow_up_chats if follow_up_chats else "No previous chat history available."}
    
    INSTRUCTIONS:
    1. Analyze the query using ONLY the provided legal context
    2. Maintain continuity with previous chats if available
    3. Structure your response clearly with:
       - Direct answer to the query
       - Key legal implications (if relevant)
       - Practical considerations (if applicable)
    4. If context is insufficient, state: "Based on the provided clauses: [answer with available information]"
    5. Be specific and avoid generic legal advice
    6. Use clear, professional language
    
    RESPONSE:
    """
    
    try:
        response = genaimodel.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1500
            }
        )
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Legal analysis unavailable due to technical error: {str(e)}"



def get_formatted_bot_response(
    user_message: str,
    db: db_dependency,
    session_id: Optional[int] = None
) -> Dict:
    """
    Generate a comprehensive formatted legal response.
    
    Args:
        user_message: User's legal query
        db: Database session
        session_id: Optional chat session ID
        
    Returns:
        Dictionary with response, intent, status, and context usage info
    """
    try:
        # Step 1: Analyze query intent
        analysis_result = analyze_query_intent_and_chunks(user_message)
        if not analysis_result or "intent" not in analysis_result:
            raise ValueError("Failed to analyze query intent")
        
        logger.info(f"Query analysis: {analysis_result}")
        
        # Step 2: Search for relevant context
        search_results = search_top_k(analysis_result, k=DEFAULT_TOP_K)
        if not search_results or search_results.get("error"):
            error_msg = search_results.get("error", "No search results") if search_results else "Search failed"
            return {
                "response": f"Unable to find relevant legal clauses: {error_msg}",
                "intent": analysis_result["intent"],
                "status": "error",
                "context_used": False
            }
        
        # Step 3: Format context
        rag_context = get_context_from_search_results(search_results)
        
        # Step 4: Get follow-up chats
        follow_up_chats = None
        if session_id:
            try:
                follow_up_chats = get_follow_up_chats(db, session_id)
                if follow_up_chats == "No follow-up chats found for this session.":
                    follow_up_chats = None
            except Exception as e:
                logger.warning(f"Could not retrieve chat history: {e}")
        
        # Step 5: Generate response
        raw_response = answer_with_gemini_rag(
            user_query=user_message,
            context=rag_context,
            follow_up_chats=follow_up_chats,
            session_id=session_id,
            db=db
        )
        
        return {
            "response": raw_response,
            "intent": analysis_result["intent"],
            "status": "success",
            "context_used": bool(rag_context and rag_context.strip())
        }
        
    except Exception as e:
        logger.error(f"Error in get_formatted_bot_response: {e}")
        return {
            "response": f"Legal analysis unavailable: {str(e)}",
            "intent": "error",
            "status": "error",
            "context_used": False
        }


def summarize_user_message(message: str) -> str:
    """
    Create a short summary of user message for logging/display.
    
    Args:
        message: Full user message
        
    Returns:
        Shortened summary
    """
    if not message:
        return "Empty message"
    
    words = message.split()
    if len(words) <= 5:
        return message
    
    return " ".join(words[:5]) + "..."
