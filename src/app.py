"""FastAPI application factory and core setup."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.database import connect_to_mongo, close_mongo_connection
from src.routers import auth, pages, notebooks, flashcards


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Study Assist API",
        description="RAG-powered study assistant API for learning from Course documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration (keep in sync with frontend expectations)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure core data directories exist early
    config.ensure_directories()

    # Include routers
    app.include_router(auth.router)
    app.include_router(pages.router)
    app.include_router(notebooks.router)
    app.include_router(flashcards.router)
    # app.include_router(rag.router)
    # app.include_router(notes.router)

    @app.get("/")
    async def root():
        """Root endpoint with basic API metadata."""
        return {
            "message": "Study Assist API",
            "version": app.version,
            "docs": "/docs",
        }

    return app


# Application instance used by ASGI servers (uvicorn, etc.)
app = create_app()



