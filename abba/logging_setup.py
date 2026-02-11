"""Centralized logging setup using loguru for ABBA."""

import sys
from typing import Optional

from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """Setup loguru logging with appropriate level and formatting.

    Args:
        log_level: The log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default loguru handler
    logger.remove()

    effective_level = log_level

    # Add custom handler with clean formatting
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=effective_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # For progress bar compatibility, add a method to write messages that don't interfere
    def tqdm_write(message: str, level: str = "INFO"):
        """Write a log message using tqdm.write() to avoid interfering with progress bars."""
        try:
            from tqdm import tqdm

            tqdm.write(f"{level}: {message}")
        except ImportError:
            # Fallback to regular logging if tqdm not available
            getattr(logger, level.lower())(message)

    # Add the tqdm_write method to logger
    logger.tqdm_write = tqdm_write


def get_logger(name: Optional[str] = None):
    """Get a logger instance.

    Args:
        name: Optional name for the logger (defaults to calling module)

    Returns:
        Configured loguru logger
    """
    if name:
        return logger.bind(name=name)
    return logger


# Compatibility function for existing code using standard logging
def configure_standard_logging():
    """Configure standard Python logging to use loguru as backend."""
    import logging

    class InterceptHandler(logging.Handler):
        """Intercept standard logging messages and route them to loguru."""

        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Replace all existing loggers with intercept handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress some noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
