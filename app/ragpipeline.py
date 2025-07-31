#Libraries
import pandas as pd
import zipfile
import os
from typing import BinaryIO
from tempfile import NamedTemporaryFile
import google.generativeai as genai
import json
import re
from db import vector_db
import pandas as pd
import uuid
from fastapi.templating import Jinja2Templates
from db_crud import get_follow_up_chats
from db import db_dependency


#LLM API's & Embedding Model Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
genaimodel=genai.GenerativeModel(model_name='gemini-2.0-flash')

templates = Jinja2Templates(directory="templates")

def embed_and_store_csv(file: BinaryIO, filename: str):
    df = pd.read_csv(file)
    
    if "clause_text" not in df.columns:
        raise ValueError("Missing 'clause_text' column.")
    
    # Check if this file was already processed
    existing = vector_db.get(
        where={"source_file": filename},  # Metadata filter
        limit=1
    )

    if existing and existing['ids']:
        raise ValueError(f" Embeddings from '{filename}' already exist.")

    df = df.dropna(subset=["clause_text"])
    texts = df['clause_text'].tolist()
    types = df.get('clause_type', ['Unknown'] * len(texts)).tolist()

    vector_db.add(
        ids=[str(uuid.uuid4()) for _ in texts],
        metadatas=[{"type": t, "source_file": filename} for t in types],
        documents=texts
    )

    return f"✅ {len(texts)} clauses from '{filename}' added successfully."


def handle_uploaded_file(file: BinaryIO, filename: str):
    """Determine file type and process accordingly."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        embed_and_store_csv(file,filename)

    elif ext == ".zip":
        # Save the uploaded zip temporarily
        with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(file.read())
            tmp.flush()

            with zipfile.ZipFile(tmp.name, "r") as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv")]

                if not csv_files:
                    raise ValueError("No CSV files found in the zip.")

                for csv_file in csv_files:
                    with zip_ref.open(csv_file) as f:
                        embed_and_store_csv(f,os.path.basename(csv_file))

            os.remove(tmp.name)  # Clean up

    else:
        raise ValueError("Only .csv and .zip files are allowed.")

def analyze_query_intent_and_chunks(user_query):
        prompt = f"""
    You are an intelligent NLP assistant. Given a user query, do the following:

    1. Think step-by-step and determine the user's intent from these options:
    - "compare": if the user is explicitly comparing two things (using words like compare, vs, difference between, better than, etc.)
    - "define": if the user wants a definition or explanation of a single concept.
    - "retrieve": if the user wants to know applications, uses, benefits, etc. of a topic.

    2. Based on the intent, extract:
    - If intent is "compare" → return two separate concepts the user wants to compare
    - If intent is "define" or "retrieve" → return one main concept only

    3. Return output ONLY in this JSON format (no extra explanation):

    {{
    "intent": "compare" | "define" | "retrieve",
    "chunks": ["first", "second"] or ["single"]
    }}

    User query:
    "{user_query}"
    """

        try:
            response = genaimodel.generate_content(prompt)
            raw_text = response.text.strip()

                    # Extract valid JSON only
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("No valid JSON found")

            # 🔒 Final structure enforcement
            if result["intent"] in ("define", "retrieve") and len(result["chunks"]) > 1:
                joined_chunk = " ".join(result["chunks"])
                result["chunks"] = [joined_chunk]

            return result
        except Exception as e:
            print("Error:", e)
            return {"intent": "unknown", "chunks": []}
        
        
def search_top_k(result, k=5):
    if not result or "chunks" not in result or not result["chunks"]:
        return {"intent": "unknown", "results": []}

    intent = result.get("intent", "unknown")
    chunks = result["chunks"]

    query_result = vector_db.query(
        query_texts=chunks,
        n_results=k
    )

    # Results will be a list if multiple chunks; single list if one chunk
    all_results = []

    for i, (metas, docs, dists) in enumerate(
        zip(
            query_result.get("metadatas", []),
            query_result.get("documents", []),
            query_result.get("distances", [])
        )
    ):
        chunk_results = []
        for meta, doc, dist in zip(metas, docs, dists):
            chunk_results.append({
                "score": float(dist),
                "clause_text": doc,
                "clause_type": meta.get("clause_type", "Unknown"),
                "source_file": meta.get("source_file", ""),
                "row_index": meta.get("row_index", -1)
            })

        # Use "results" label for single-chunk intents
        if intent in ("define", "retrieve") or len(chunks) == 1:
            return {
                "intent": intent,
                "chunks": chunks,
                "results": chunk_results
            }

        # Else handle compare as list of labeled results
        label = f"chunk_{i+1}"
        all_results.append({label: chunk_results})

    return {
        "intent": intent,
        "chunks": chunks,
        "results": all_results
    }



def format_rag_context(retrieval: dict) -> str:
    intent = retrieval.get("intent", "unknown")
    results = retrieval.get("results", [])
    context = ""

    if not results:
        return "No relevant clauses found."

    # Handle "define" or "retrieve" (single chunk result)
    if intent in ("define", "retrieve") and isinstance(results, list) and isinstance(results[0], dict) and "clause_text" in results[0]:
        context += "### Retrieved Legal Clauses:\n"
        for i, m in enumerate(results, 1):
            context += f"{i}. {m['clause_text'].strip()}"
            if m.get("clause_type"):
                context += f" (Type: {m['clause_type']})"
            if m.get("source_file"):
                context += f" — from {m['source_file']}"
            context += "\n"
        return context.strip()

    # Handle "compare" (multiple chunks)
    for entry in results:
        for label, matches in entry.items():
            context += f"\n### Results for {label.replace('_', ' ').title()}:\n"
            for i, m in enumerate(matches, 1):
                context += f"{i}. {m['clause_text'].strip()}"
                if m.get("clause_type"):
                    context += f" (Type: {m['clause_type']})"
                if m.get("source_file"):
                    context += f" — from {m['source_file']}"
                context += "\n"

    return context.strip()

def answer_with_gemini_rag(
    user_query: str, 
    context: str,
    follow_up_chats: str = None,
    session_id: int = None,
    db: db_dependency = None
) -> str:
    """
    Generates a legal answer using Gemini with RAG (Retrieval-Augmented Generation).
    
    Args:
        user_query: The user's legal question
        context: Relevant legal clauses for grounding
        follow_up_chats: Previous chat history (optional)
        session_id: Current session ID (optional)
        db: Database dependency (optional)
        
    Returns:
        Markdown-formatted legal answer with citations
    """
    # Enhanced follow-up chat retrieval if session_id provided
    if session_id and db and not follow_up_chats:
        follow_up_chats = get_follow_up_chats(db, session_id)
    
    # Structured prompt engineering
      # Structured prompt engineering
    prompt = f"""
    LEGAL ANALYSIS REQUEST
    ----------------------

    ROLE:
    You are a senior legal AI assistant trained in contract law, multilingual greetings, and user-friendly communication. Begin interactions professionally and politely. If the user's message is a greeting or salutation (e.g., "Hi", "Hello", or greetings in Urdu or other languages), greet them back in the same language if possible, introduce yourself, and ask how you may assist them legally.

    CONTEXT (Relevant Clauses):
    {context}
    
    CURRENT QUERY:
    {user_query}
    
    CHAT HISTORY:
    {follow_up_chats if follow_up_chats else "No previous chat history available"}
    
    INSTRUCTIONS:
    1. If the user’s input is a greeting or introduction, respond politely and introduce yourself.
    2. Otherwise, analyze the query STRICTLY within the provided legalclauses context.
    3. Maintain continuity with any previous chats.
    4. Structure your response as:
       - A direct answer to the user’s query
       - Potential implications (if applicable)
    5. Use clean and clear Markdown formatting.
    6. If legal context is insufficient to answer the query, reply: "Based on the provided clauses: [Partial answer]"
    7. Do NOT provide generalized or irrelevant legal information.

    RESPONSE:
    """

    
    try:
        # Generate with safety settings
        response = genaimodel.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # More deterministic
                "max_output_tokens": 1500
            },
            safety_settings={
                "HARM_CATEGORY_DANGEROUS": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE"
            }
        )
        
        # Post-processing
        return response.text
    except Exception as e:
        return f"## Legal Analysis Unavailable\nError: {str(e)}"

def get_formatted_bot_response(
    user_message: str,
    db: db_dependency,
    session_id: int = None
) -> dict:
    """
    Generates a formatted legal response with context-aware RAG.
    
    Args:
        user_message: User's legal query
        db: Database session
        session_id: Optional chat session ID
        
    Returns:
        {
            "response": HTML-formatted answer,
            "intent": detected intent,
            "status": "success"|"error",
            "context_used": bool  # Whether context was utilized
        }
    """
    try:
        # ===== 1. Query Analysis =====
        analysis_result = analyze_query_intent_and_chunks(user_message)
        if not analysis_result or "intent" not in analysis_result:
            raise ValueError("Failed to analyze query intent")
        
        # ===== 2. Context Retrieval =====
        search_results = search_top_k(analysis_result, k=10)
        if not search_results:
            return {
                "response": "<p>No relevant legal clauses found.</p>",
                "intent": analysis_result["intent"],
                "status": "success",
                "context_used": False
            }
        
        # ===== 3. Context Formatting =====
        rag_context = format_rag_context(search_results)
        
        # ===== 4. Follow-up Chats =====
        follow_up_chats = None
        if session_id:
            follow_up_chats = get_follow_up_chats(db, session_id)
            if follow_up_chats == "No follow-up chats found for this session.":
                follow_up_chats = None  # Treat as no history
        
        # ===== 5. Generate Response =====
        response = answer_with_gemini_rag(
            user_query=user_message,
            context=rag_context,
            follow_up_chats=follow_up_chats
        )
        
        # ===== 6. Formatting & Return =====
        return {
            "response": response,
            "intent": analysis_result["intent"],
            "status": "success",
            "context_used": bool(rag_context.strip())
        }
        
    except Exception as e:
        error_msg = f"<p class='error'>Legal analysis unavailable: {str(e)}</p>"
        return {
            "response": error_msg,
            "intent": "error",
            "status": "error",
            "context_used": False
        }

def summarize_user_message(message: str) -> str:
    words = message.split()
    short_summary = " ".join(words[:5]) + "..." if len(words) > 5 else message
    return short_summary