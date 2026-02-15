"""FastAPI application factory for ABBA."""

from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import configure_db, router


def create_app(db_path: Optional[Union[str, Path]] = None) -> FastAPI:
    """Create and configure the ABBA FastAPI application.

    Args:
        db_path: Optional path to a pre-existing ABBA SQLite database.
                 If provided, the routes will be configured with this database
                 instead of using the default config-based lookup.

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

    if db_path is not None:
        from ..database import SQLiteManager

        db_manager = SQLiteManager(Path(db_path))
        configure_db(db_manager)

    return app
