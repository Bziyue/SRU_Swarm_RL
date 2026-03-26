from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    return PROJECT_DIR.joinpath(*parts)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_DIR / candidate
