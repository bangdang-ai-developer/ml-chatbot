from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from .api.chat import router as chat_router
from .api.webui import router as webui_router
from .core.config import settings
from .core.exceptions import ChatbotException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Chatbot API",
    description="Chatbot for Andrew Ng's Machine Learning Yearning book",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(webui_router, prefix="/v1", tags=["webui"])

# Exception handlers
@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request, exc):
    logger.error(f"Chatbot exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ML Chatbot API",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Chatbot API for Andrew Ng's Machine Learning Yearning",
        "docs": "/docs",
        "health": "/health"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting ML Chatbot API...")

    try:
        # Validate configuration
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ML Chatbot API...")

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )