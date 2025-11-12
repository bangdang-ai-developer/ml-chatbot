from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import time
import uuid
import logging
from ..models.webui import (
    WebUIChatRequest,
    WebUIChatResponse,
    WebUIChoice,
    WebUIUsage,
    WebUIMessage,
    WebUIModelInfo,
    WebUIModelsResponse
)
from ..services.rag_service import RAGService
from ..services.ai_service import GeminiAIService
from ..services.smart_response_router import SmartResponseRouter
from ..repositories.vector_repository import MilvusRepository
from ..services.embedding_service import GeminiEmbeddingService
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency injection
async def get_intelligent_agent() -> SmartResponseRouter:
    """Get intelligent agent with smart routing"""
    try:
        logger.info("Initializing intelligent agent for WebUI...")

        # Initialize core services
        vector_repo = MilvusRepository()
        embedding_service = GeminiEmbeddingService()
        ai_service = GeminiAIService()

        # Initialize RAG service
        rag_service = RAGService(vector_repo, embedding_service, ai_service)

        # Initialize smart response router
        smart_router = SmartResponseRouter(rag_service, ai_service)

        logger.info("Intelligent agent for WebUI initialized successfully")
        return smart_router

    except Exception as e:
        logger.error(f"Failed to initialize intelligent agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize chatbot services")

@router.post("/chat/completions")
async def chat_completions(
    request: WebUIChatRequest,
    smart_agent: SmartResponseRouter = Depends(get_intelligent_agent)
) -> Dict[str, Any]:
    """Chat completions endpoint for Open WebUI - Gemini-2.5-Flash only"""
    try:
        logger.info(f"Received chat completion request with {len(request.messages)} messages")

        # Extract the latest user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in request")

        # Generate intelligent response using smart routing (same as /api/v1/chat)
        response_result = await smart_agent.generate_intelligent_response(
            query=user_message,
            query_analysis={},  # Could be enhanced with query analysis
            session_id=request.session_id or "webui-default"
        )

        # Create response in OpenAI-compatible format
        completion_id = str(uuid.uuid4())
        created_time = int(time.time())

        response_message = {
            "role": "assistant",
            "content": response_result.response
        }

        choice = {
            "index": 0,
            "message": response_message,
            "finish_reason": "stop"
        }

        usage = {
            "prompt_tokens": len(user_message.split()) * 4,  # Rough estimate
            "completion_tokens": len(response_result.response.split()) * 4,  # Rough estimate
            "total_tokens": len(user_message.split()) * 4 + len(response_result.response.split()) * 4
        }

        webui_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "gemini-2.5-flash",  # Always use our single model
            "choices": [choice],
            "usage": usage
        }

        # Completely remove sources and confidence to test if this fixes frontend errors
        # Open WebUI might have issues with custom formatting
        # webui_response["sources"] = rag_response.sources if rag_response.sources else []
        # webui_response["confidence"] = rag_response.confidence if rag_response.confidence is not None else None

        # Explicitly remove any sources or confidence that might have been added elsewhere
        webui_response.pop("sources", None)
        webui_response.pop("confidence", None)

        logger.info(f"Successfully processed chat completion: {user_message[:50]}... using strategy: {response_result.strategy_used}")
        return webui_response

    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=WebUIModelsResponse)
async def list_models():
    """List available models - Gemini-2.5-Flash only"""
    try:
        models = [
            WebUIModelInfo(
                id="gemini-2.5-flash",
                created=int(time.time()),
                owned_by="google",
                description="Google Gemini 2.5 Flash - Fast and efficient model for general chat with RAG capabilities"
            )
        ]

        return WebUIModelsResponse(data=models)

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model information"""
    try:
        if model_id != "gemini-2.5-flash":
            raise HTTPException(status_code=404, detail="Model not found")

        model_info = WebUIModelInfo(
            id="gemini-2.5-flash",
            created=int(time.time()),
            owned_by="google",
            description="Google Gemini 2.5 Flash - Fast and efficient model for general chat with RAG capabilities"
        )

        return model_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/completions")
async def stream_chat_completions(
    request: WebUIChatRequest,
    smart_agent: SmartResponseRouter = Depends(get_intelligent_agent)
):
    """Streaming chat completions (optional implementation)"""
    # For now, return non-streaming response
    # Can implement streaming later if needed
    return await chat_completions(request, smart_agent)

@router.get("/health")
async def health_check():
    """Health check endpoint for Open WebUI"""
    return {
        "status": "healthy",
        "service": "ML Chatbot WebUI API",
        "model": "gemini-2.5-flash",
        "version": "1.0.0"
    }

