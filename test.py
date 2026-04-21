"""Backward-compatible entry: ``uvicorn test:app`` or ``python test.py``."""

from speech_app.api import app, run

__all__ = ["app", "run"]

if __name__ == "__main__":
    run()
