"""FastAPI application factory for ABBA."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    """Create and configure the ABBA FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="ABBA Bible Study API",
        description=(
            "Annotated Bible and Background Analysis — "
            "making scholar-level biblical knowledge accessible to everyday readers."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
