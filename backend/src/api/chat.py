from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any
import logging
from ..models.chat import ChatRequest, ChatResponse
from ..services.rag_service import RAGService
from ..services.document_service import PDFDocumentService
from ..services.embedding_service import GeminiEmbeddingService
from ..services.ai_service import GeminiAIService
from ..services.smart_response_router import SmartResponseRouter
from ..services.query_expansion_service import QueryExpansionService
from ..services.vietnamese_text_service import VietnameseTextService
from ..repositories.vector_repository import MilvusRepository
from ..core.config import settings
from ..core.exceptions import ChatbotException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
async def get_rag_service() -> RAGService:
    """Get RAG service instance"""
    try:
        # Initialize core services
        vector_repo = MilvusRepository()
        embedding_service = GeminiEmbeddingService()
        ai_service = GeminiAIService()

        # Initialize RAG service
        rag_service = RAGService(vector_repo, embedding_service, ai_service)

        return rag_service

    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize RAG service")

async def get_intelligent_agent() -> SmartResponseRouter:
    """Get intelligent agent with smart routing"""
    try:
        logger.info("Initializing intelligent agent with smart routing...")

        # Get RAG service first
        rag_service = await get_rag_service()

        # Initialize smart response router
        smart_router = SmartResponseRouter(rag_service, rag_service.ai_service)

        logger.info("Intelligent agent initialized successfully")
        return smart_router

    except Exception as e:
        logger.error(f"Failed to initialize intelligent agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize intelligent chatbot services")

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    smart_agent: SmartResponseRouter = Depends(get_intelligent_agent)
):
    """Chat with the intelligent ML chatbot"""
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        # Generate intelligent response using smart routing
        response_result = await smart_agent.generate_intelligent_response(
            query=request.message,
            query_analysis={},  # Could be enhanced with query analysis
            session_id=request.session_id or "default"
        )

        # Convert to ChatResponse format
        chat_response = ChatResponse(
            message=response_result.response,
            session_id=request.session_id or "default",
            sources=response_result.sources if response_result.sources else None,
            confidence=response_result.confidence
        )

        # Log routing information
        logger.info(f"Response generated using strategy: {response_result.strategy_used}")
        logger.info(f"Processing time: {response_result.processing_time:.2f}s")

        return chat_response

    except ChatbotException as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in intelligent chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/index-document")
async def index_document(
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Index the ML Yearning PDF document"""
    try:
        # Run indexing in background
        background_tasks.add_task(index_document_task, rag_service)

        return {"message": "Document indexing started in background"}

    except Exception as e:
        logger.error(f"Failed to start document indexing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start document indexing")

@router.get("/indexing-status")
async def get_indexing_status():
    """Get document indexing status"""
    try:
        # This would typically check a database or cache for indexing status
        # For now, return a simple status
        return {
            "status": "completed",  # or "in_progress", "failed"
            "indexed_chunks": 0,  # Would be actual count
            "last_updated": None
        }

    except Exception as e:
        logger.error(f"Failed to get indexing status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get indexing status")

@router.get("/stats")
async def get_chatbot_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get chatbot statistics"""
    try:
        stats = await rag_service.get_document_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get chatbot stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chatbot stats")

@router.post("/reindex")
async def reindex_documents(
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Reindex all documents (clear and reindex)"""
    try:
        # Run reindexing in background
        background_tasks.add_task(reindex_task, rag_service)

        return {"message": "Document reindexing started in background"}

    except Exception as e:
        logger.error(f"Failed to start document reindexing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start document reindexing")

# Background tasks
async def index_document_task(rag_service: RAGService):
    """Background task to index multiple PDF documents"""
    try:
        logger.info("Starting multi-document indexing...")

        # Initialize document service
        doc_service = PDFDocumentService()

        all_document_chunks = []

        # Process each PDF
        for i, pdf_path in enumerate(settings.pdf_paths):
            try:
                pdf_name = pdf_path.split('/')[-1]
                logger.info(f"Processing {pdf_name} ({i+1}/{len(settings.pdf_paths)})...")

                # Extract text from PDF
                logger.info(f"Extracting text from {pdf_name}...")
                text = await doc_service.extract_text_from_pdf(pdf_path)

                # Chunk the text
                logger.info(f"Chunking text from {pdf_name}...")
                text_chunks = await doc_service.chunk_text(text)

                # Create document chunks with source info
                logger.info(f"Creating {len(text_chunks)} document chunks from {pdf_name}...")
                document_chunks = await doc_service.create_document_chunks(text_chunks, source=pdf_name)

                all_document_chunks.extend(document_chunks)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        # Index all documents
        if all_document_chunks:
            logger.info(f"Indexing {len(all_document_chunks)} total document chunks in vector store...")
            success = await rag_service.index_document(all_document_chunks)

            if success:
                logger.info(f"Successfully indexed {len(all_document_chunks)} document chunks from all PDFs")
            else:
                logger.error("Failed to index documents")
        else:
            logger.warning("No document chunks were created")

    except Exception as e:
        logger.error(f"Multi-document indexing failed: {e}")

async def reindex_task(rag_service: RAGService):
    """Background task to reindex all documents"""
    try:
        logger.info("Starting multi-document reindexing...")

        # Initialize document service
        doc_service = PDFDocumentService()

        all_document_chunks = []

        # Process each PDF
        for i, pdf_path in enumerate(settings.pdf_paths):
            try:
                pdf_name = pdf_path.split('/')[-1]
                logger.info(f"Processing {pdf_name} ({i+1}/{len(settings.pdf_paths)})...")

                # Extract text from PDF
                logger.info(f"Extracting text from {pdf_name}...")
                text = await doc_service.extract_text_from_pdf(pdf_path)

                # Chunk the text
                logger.info(f"Chunking text from {pdf_name}...")
                text_chunks = await doc_service.chunk_text(text)

                # Create document chunks with source info
                logger.info(f"Creating {len(text_chunks)} document chunks from {pdf_name}...")
                document_chunks = await doc_service.create_document_chunks(text_chunks, source=pdf_name)

                all_document_chunks.extend(document_chunks)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        # Reindex all documents
        if all_document_chunks:
            logger.info(f"Reindexing {len(all_document_chunks)} total document chunks...")
            success = await rag_service.reindex_all_documents(all_document_chunks)

            if success:
                logger.info(f"Successfully reindexed {len(all_document_chunks)} document chunks from all PDFs")
            else:
                logger.error("Failed to reindex documents")
        else:
            logger.warning("No document chunks were created for reindexing")

    except Exception as e:
        logger.error(f"Multi-document reindexing failed: {e}")