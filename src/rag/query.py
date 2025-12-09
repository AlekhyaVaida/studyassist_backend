"""RAG query service for retrieving relevant documents."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config
from src.rag.store import get_user_vector_store


def query_rag(
    user_id: str,
    query: str,
    notebook_id: Optional[str] = None,
    k: int = None,
    return_raw_chunks: bool = False
) -> Dict[str, Any]:
    """Query RAG system for relevant documents.
    
    Args:
        user_id: User ID to scope the search
        query: Query text
        notebook_id: Optional notebook ID to filter results
        k: Number of results to retrieve
        return_raw_chunks: If True, return raw chunks without LLM processing (faster)
        
    Returns:
        dict with 'answer' and 'sources' keys, or 'chunks' if return_raw_chunks=True
    """
    # Get user's vector store
    vector_store = get_user_vector_store(user_id)
    
    # Build filter
    filter_dict = {}
    if notebook_id:
        filter_dict["notebook_id"] = notebook_id
    
    # Search for relevant chunks
    results = vector_store.search(query, k=k, filter_dict=filter_dict)
    
    if not results:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks": []
        }
    
    # If return_raw_chunks, skip LLM processing for speed
    if return_raw_chunks:
        return {
            "chunks": [r["text"] for r in results],
            "sources": [
                {
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "metadata": r["metadata"],
                    "score": r["score"]
                }
                for r in results
            ]
        }
    
    # Combine retrieved context
    context = "\n\n".join([
        f"[Chunk {i+1}]: {r['text']}"
        for i, r in enumerate(results)
    ])
    
    # Generate answer using LLM (faster with lower temperature)
    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=0.3,  # Lower temperature for faster responses
        openai_api_key=config.openai_api_key
    )
    
    messages = [
        SystemMessage(content="""Extract relevant information from the context to answer the question.
Be concise and focus on the most relevant details."""),
        HumanMessage(content=f"""Context:
{context}

Question: {query}

Provide a concise answer based on the context:""")
    ]
    
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "sources": [
            {
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "metadata": r["metadata"],
                "score": r["score"]
            }
            for r in results
        ],
        "chunks": [r["text"] for r in results]
    }

