import logging
import sys
import uuid


def get_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def new_trace_id(prefix="req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
