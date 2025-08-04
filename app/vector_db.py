# vector_db.py - Enhanced FAISS-based Vector Database

import faiss
import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from sentence_transformers import SentenceTransformer

# Import settings (assuming redisconfig contains path settings)
try:
    from redisconfig import settings
except ImportError:
    # Fallback settings if redisconfig is not available
    class Settings:
        VECTOR_DB_PATH = "./RAGembeddings"  # Default path for vector DB
    settings = Settings()

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILENAME = "faiss_index.index"
METADATA_FILENAME = "faiss_metadata.pkl"

# Global variables for lazy loading
_faiss_index = None
_metadata = None
_model = None


def get_model():
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Loaded sentence transformer model: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load model {MODEL_NAME}: {e}")
            raise
    return _model


def ensure_vector_db_directory() -> bool:
    """Ensure the vector database directory exists."""
    try:
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        logger.info(f"Vector DB directory ensured at: {os.path.abspath(settings.VECTOR_DB_PATH)}")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector DB directory: {e}")
        return False


def get_file_paths() -> tuple:
    """Get the full paths for index and metadata files."""
    index_path = os.path.join(settings.VECTOR_DB_PATH, INDEX_FILENAME)
    metadata_path = os.path.join(settings.VECTOR_DB_PATH, METADATA_FILENAME)
    return index_path, metadata_path


def load_or_create_index() -> tuple:
    """
    Lazy load FAISS index and metadata.
    
    Returns:
        Tuple of (faiss_index, metadata_dict)
    """
    '''global _faiss_index, _metadata
    
    if _faiss_index is not None and _metadata is not None:
        return _faiss_index, _metadata
    '''
    ensure_vector_db_directory()
    index_path, metadata_path = get_file_paths()
    
    try:
        # Try to load existing index and metadata
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            logger.info("Loading existing FAISS index...")
            
            _faiss_index = faiss.read_index(index_path)
            
            with open(metadata_path, "rb") as f:
                _metadata = pickle.load(f)
            
            # Validate loaded data
            if not isinstance(_metadata, dict):
                logger.warning("Invalid metadata format, creating new index")
                raise ValueError("Invalid metadata format")
            
            logger.info(f"Loaded FAISS index with {_faiss_index.ntotal} vectors and {len(_metadata)} metadata entries")
            
        else:
            logger.info("Creating new FAISS index...")
            _faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            _metadata = {}
            save_faiss_index()  # Save empty index
            
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        # Fallback to new index
        logger.info("Creating fallback FAISS index...")
        _faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        _metadata = {}
        
    return _faiss_index, _metadata


def save_faiss_index() -> bool:
    """
    Persist FAISS index and metadata to disk.
    
    Returns:
        True if successful, False otherwise
    """
    global _faiss_index, _metadata
    
    if _faiss_index is None or _metadata is None:
        logger.error("Cannot save: index or metadata is None")
        return False
    
    try:
        ensure_vector_db_directory()
        index_path, metadata_path = get_file_paths()
        
        # Save FAISS index
        faiss.write_index(_faiss_index, index_path)
        
        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(_metadata, f)
        
        logger.info(f"FAISS index saved successfully. Total vectors: {_faiss_index.ntotal}")
        logger.debug(f"Files saved to: {os.path.abspath(settings.VECTOR_DB_PATH)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        return False


def add(ids: List[str], documents: List[str], metadatas: List[Dict]) -> bool:
    """
    Add text documents and metadata to FAISS + metadata store.
    
    Args:
        ids: List of unique document IDs
        documents: List of document texts
        metadatas: List of metadata dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    global _faiss_index, _metadata
    
    # Validate inputs
    if not ids or not documents or not metadatas:
        logger.error("Empty inputs provided to add()")
        return False
    
    if not (len(ids) == len(documents) == len(metadatas)):
        logger.error("Mismatched lengths in add() inputs")
        return False
    
    # Ensure index is loaded
    _faiss_index, _metadata = load_or_create_index()
    model = get_model()
    
    try:
        # Get starting index BEFORE adding new vectors
        start_index = _faiss_index.ntotal
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Validate embedding dimensions
        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {embeddings.shape[1]}")
        
        # Add to FAISS index
        _faiss_index.add(embeddings)
        
        # Update metadata with correct indices
        for i, doc_id in enumerate(ids):
            faiss_idx = start_index + i
            _metadata[faiss_idx] = {
                "id": str(doc_id),
                "text": str(documents[i]),
                "metadata": dict(metadatas[i])  # Ensure it's a dict
            }
        
        # Save to disk
        success = save_faiss_index()
        if success:
            logger.info(f"Successfully added {len(documents)} documents. Total vectors: {_faiss_index.ntotal}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error adding documents to FAISS: {e}")
        return False


def get(where: Optional[Dict] = None, limit: int = 10) -> Dict:
    """
    Get documents matching metadata criteria.
    
    Args:
        where: Dictionary of metadata criteria to match (optional)
        limit: Maximum number of results to return
        
    Returns:
        Dictionary with 'ids' and 'documents' keys
    """
    global _faiss_index, _metadata
    
    # Ensure index is loaded
    _faiss_index, _metadata = load_or_create_index()
    
    if not _metadata:
        return {"ids": [], "documents": [], "metadatas": []}
    
    try:
        results = []
        
        for faiss_idx, entry in _metadata.items():
            # If no where clause, include all
            if where is None:
                results.append(entry)
            else:
                # Check if entry matches all criteria in where clause
                entry_metadata = entry.get("metadata", {})
                if all(entry_metadata.get(k) == v for k, v in where.items()):
                    results.append(entry)
            
            # Stop if we've reached the limit
            if len(results) >= limit:
                break
        
        logger.debug(f"Retrieved {len(results)} documents matching criteria")
        
        return {
            "ids": [r.get("id", "") for r in results],
            "documents": [r.get("text", "") for r in results],
            "metadatas": [r.get("metadata", {}) for r in results]
        }
        
    except Exception as e:
        logger.error(f"Error in get(): {e}")
        return {"ids": [], "documents": [], "metadatas": []}


def similarity_search(query_text: str, n_results: int = 5) -> Dict:
    """
    Perform semantic similarity search using FAISS.
    
    Args:
        query_text: Text to search for
        n_results: Number of results to return
        
    Returns:
        Dictionary with documents, metadatas, and distances
    """
    global _faiss_index, _metadata
    
    # Ensure index is loaded
    _faiss_index, _metadata = load_or_create_index()
    
    if _faiss_index.ntotal == 0:
        logger.warning("FAISS index is empty!")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    if not query_text or not query_text.strip():
        logger.warning("Empty query text provided")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    try:
        model = get_model()
        
        # Encode query
        query_embedding = model.encode([query_text.strip()], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Perform search
        n_results = min(n_results, _faiss_index.ntotal)
        distances, indices = _faiss_index.search(query_embedding, n_results)
        
        # Extract results
        documents = []
        metadatas = []
        result_distances = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx in _metadata:
                entry = _metadata[idx]
                documents.append(entry.get("text", ""))
                metadatas.append(entry.get("metadata", {}))
                result_distances.append(float(dist))
            else:
                logger.warning(f"Invalid index {idx} found in search results")
        
        logger.info(f"Similarity search completed. Found {len(documents)} results for query: '{query_text[:50]}...'")
        
        return {
            "documents": [documents],  # Nested list format for compatibility
            "metadatas": [metadatas],
            "distances": [result_distances]
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the current index.
    
    Returns:
        Dictionary with index statistics
    """
    global _faiss_index, _metadata
    
    try:
        _faiss_index, _metadata = load_or_create_index()
        index_path, metadata_path = get_file_paths()
        
        stats = {
            "total_vectors": _faiss_index.ntotal if _faiss_index else 0,
            "metadata_entries": len(_metadata) if _metadata else 0,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "files_exist": {
                "index": os.path.exists(index_path),
                "metadata": os.path.exists(metadata_path)
            },
            "model_loaded": _model is not None,
            "embedding_dimension": EMBEDDING_DIMENSION
        }
        
        logger.debug(f"Index stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return {
            "total_vectors": 0,
            "metadata_entries": 0,
            "error": str(e)
        }


def delete_by_source_file(source_file: str) -> int:
    """
    Delete documents by source file.
    
    Args:
        source_file: Name of the source file to delete documents from
        
    Returns:
        Number of documents deleted
    """
    global _faiss_index, _metadata
    
    _faiss_index, _metadata = load_or_create_index()
    
    if not _metadata:
        return 0
    
    try:
        # Find indices to delete
        indices_to_delete = []
        for faiss_idx, entry in _metadata.items():
            if entry.get("metadata", {}).get("source_file") == source_file:
                indices_to_delete.append(faiss_idx)
        
        if not indices_to_delete:
            logger.info(f"No documents found for source file: {source_file}")
            return 0
        
        # Remove from metadata
        for idx in indices_to_delete:
            del _metadata[idx]
        
        # Note: FAISS doesn't support efficient deletion, so we would need to rebuild
        # For now, we just remove from metadata (which is sufficient for most use cases)
        logger.warning("FAISS index not rebuilt - deleted entries still in vector space")
        
        # Save updated metadata
        save_faiss_index()
        
        logger.info(f"Deleted {len(indices_to_delete)} documents from source file: {source_file}")
        return len(indices_to_delete)
        
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        return 0


def rebuild_index() -> bool:
    """
    Rebuild the FAISS index from current metadata (useful after deletions).
    
    Returns:
        True if successful, False otherwise
    """
    global _faiss_index, _metadata
    
    try:
        _faiss_index, _metadata = load_or_create_index()
        
        if not _metadata:
            logger.info("No metadata to rebuild from")
            return True
        
        logger.info(f"Rebuilding index from {len(_metadata)} entries...")
        
        # Extract data from metadata
        documents = []
        metadatas = []
        ids = []
        
        for entry in _metadata.values():
            documents.append(entry.get("text", ""))
            metadatas.append(entry.get("metadata", {}))
            ids.append(entry.get("id", ""))
        
        # Create new index
        _faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        _metadata = {}
        
        # Re-add all documents
        success = add(ids, documents, metadatas)
        
        if success:
            logger.info("Index rebuilt successfully")
        
        return success
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return False


def clear_database() -> bool:
    """
    Clear all data from the vector database.
    
    Returns:
        True if successful, False otherwise
    """
    global _faiss_index, _metadata
    
    try:
        _faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        _metadata = {}
        
        success = save_faiss_index()
        
        if success:
            logger.info("Vector database cleared successfully")
        
        return success
        
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False