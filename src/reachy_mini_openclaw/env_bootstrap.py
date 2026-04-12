"""Load `.env` for editable installs and for venv/site-packages installs."""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_dotenv_for_app(*, override: bool = False) -> None:
    """Prefer `.env` walking up from cwd, then repo-style layout (editable: …/src/reachy_mini_openclaw/)."""
    cwd_file = find_dotenv(usecwd=True)
    fallback = Path(__file__).resolve().parent.parent.parent / ".env"
    if cwd_file:
        load_dotenv(cwd_file, override=override)
    elif fallback.exists():
        load_dotenv(fallback, override=override)
