"""Vector store for RAG with user-scoped embeddings."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss-cpu not installed. Vector store will use in-memory storage.")

from src.config import config


class UserVectorStore:
    """Per-user vector store for document embeddings."""
    
    def __init__(self, user_id: str):
        """Initialize vector store for a specific user.
        
        Args:
            user_id: User ID to scope the vector store
        """
        self.user_id = user_id
        self.store_dir = config.data_dir / "vector_stores" / user_id
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.store_dir / "faiss_index.bin"
        self.metadata_path = self.store_dir / "metadata.pkl"
        
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        
        # Initialize or load index
        self.index = None
        self.metadata = []
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required for vector store. Install it with: pip install faiss-cpu")
        
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded vector store for user {self.user_id} with {len(self.metadata)} chunks")
            except Exception as e:
                print(f"Error loading vector store: {e}, creating new one")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new empty FAISS index."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required for vector store. Install it with: pip install faiss-cpu")
        
        # Create index with dimension matching embedding model (1536 for text-embedding-3-small)
        dimension = 1536  # Default for text-embedding-3-small
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        print(f"Created new vector store for user {self.user_id}")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dicts for each chunk
        """
        if not texts:
            return
        
        # Split texts into chunks if needed
        all_chunks = []
        all_metadatas = []
        
        for i, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            all_chunks.extend(chunks)
            
            # Add metadata for each chunk
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            for chunk in chunks:
                chunk_metadata = {
                    **base_metadata,
                    "user_id": self.user_id,
                    "chunk_index": len(all_metadatas),
                }
                all_metadatas.append(chunk_metadata)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings_list = self.embeddings.embed_documents(all_chunks)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Add to index
        if self.index.ntotal == 0:
            # First addition - ensure dimension matches
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, metadata in enumerate(all_metadatas):
            metadata["text"] = all_chunks[i]
            self.metadata.append(metadata)
        
        # Save index and metadata
        self._save()
        
        print(f"Added {len(all_chunks)} chunks to vector store for user {self.user_id}")
    
    def search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return (defaults to config.top_k_retrieval)
            filter_dict: Optional metadata filters (e.g., {"notebook_id": "..."})
            
        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        if self.index.ntotal == 0:
            return []
        
        k = k or config.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, min(k * 2, self.index.ntotal))
        
        # Filter results by metadata
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Apply filters
                if filter_dict:
                    match = all(
                        metadata.get(key) == value
                        for key, value in filter_dict.items()
                    )
                    if not match:
                        continue
                
                results.append({
                    "text": metadata.get("text", ""),
                    "metadata": {k: v for k, v in metadata.items() if k != "text"},
                    "score": float(distance)
                })
                
                if len(results) >= k:
                    break
        
        return results
    
    def _save(self):
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def delete_by_notebook(self, notebook_id: str):
        """Delete all chunks associated with a notebook.
        
        Args:
            notebook_id: Notebook ID to delete chunks for
        """
        # Filter out metadata for this notebook
        original_count = len(self.metadata)
        self.metadata = [
            m for m in self.metadata
            if m.get("notebook_id") != notebook_id
        ]
        
        removed_count = original_count - len(self.metadata)
        
        if removed_count > 0:
            # Rebuild index
            self._rebuild_index()
            print(f"Removed {removed_count} chunks for notebook {notebook_id}")
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current metadata."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required for vector store. Install it with: pip install faiss-cpu")
        
        if not self.metadata:
            self._create_new_index()
            return
        
        # Extract texts and regenerate embeddings
        texts = [m["text"] for m in self.metadata]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create new index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        # Save
        self._save()


def get_user_vector_store(user_id: str) -> UserVectorStore:
    """Get or create vector store for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        UserVectorStore instance
    """
    return UserVectorStore(user_id)

