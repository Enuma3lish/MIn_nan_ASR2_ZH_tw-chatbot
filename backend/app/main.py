from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys

from .config import settings
from .api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("=" * 80)
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"ReDoc Documentation: http://localhost:8000/redoc")
    logger.info("=" * 80)

    # Optionally preload models on startup (can be slow)
    # Uncomment the following lines to preload models:
    # try:
    #     from .services.asr_service import asr_service
    #     from .services.tts_service import tts_service
    #     logger.info("Preloading AI models...")
    #     asr_service.load_models()
    #     tts_service.load_models()
    #     logger.info("AI models preloaded successfully")
    # except Exception as e:
    #     logger.warning(f"Could not preload models: {str(e)}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Min Nan & Chinese Voice Chatbot API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }


# Include API routes
app.include_router(router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
