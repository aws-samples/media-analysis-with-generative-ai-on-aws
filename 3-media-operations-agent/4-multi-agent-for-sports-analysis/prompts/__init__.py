"""
Prompt Templates for Media Operations Multiagent System

This module provides centralized prompt management for all agents.
"""

from .news_prompts import NewsPrompts
from .metadata_prompts import MetadataPrompts
from .requirements_prompts import RequirementsPrompts
from .qc_prompts import QCPrompts
from .film_prompts import FilmPrompts
from .sports_prompts import SportsPrompts
from .orchestrator_prompts import OrchestratorPrompts
from .base_prompts import PromptManager, prompt_manager

__all__ = [
    'NewsPrompts',
    'MetadataPrompts', 
    'RequirementsPrompts',
    'QCPrompts',
    'FilmPrompts',
    'SportsPrompts',
    'OrchestratorPrompts',
    'PromptManager',
    'prompt_manager'
]