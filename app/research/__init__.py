"""
Research and source discovery package for DEVSCOUT.
"""
from app.research.github import GitHubConnector
from app.research.models import ResearchDocument, ResearchError

__all__ = [
    "GitHubConnector",
    "ResearchDocument",
    "ResearchError",
]
