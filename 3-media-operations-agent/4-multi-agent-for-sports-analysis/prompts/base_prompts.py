"""
Base Prompt Classes and Utilities

Common functionality and utilities for prompt management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
from datetime import datetime


class BasePromptTemplate(ABC):
    """Base class for prompt templates with common functionality."""
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        pass
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with parameters."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for prompt template: {e}")
    
    def validate_parameters(self, required_params: list, provided_params: dict) -> bool:
        """Validate that all required parameters are provided."""
        missing_params = [param for param in required_params if param not in provided_params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        return True


class PromptManager:
    """Central manager for all prompt templates."""
    
    def __init__(self):
        self._templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates."""
        from .news_prompts import NewsPrompts
        from .metadata_prompts import MetadataPrompts
        from .requirements_prompts import RequirementsPrompts
        from .qc_prompts import QCPrompts
        from .orchestrator_prompts import OrchestratorPrompts
        from .sports_prompts import SportsPrompts
        from .film_prompts import FilmPrompts
        
        self._templates = {
            'news': NewsPrompts,
            'metadata': MetadataPrompts,
            'requirements': RequirementsPrompts,
            'qc': QCPrompts,
            'orchestrator': OrchestratorPrompts,
            'sports': SportsPrompts,
            'film': FilmPrompts
        }
    
    def get_template(self, agent_type: str):
        """Get prompt template for a specific agent type."""
        if agent_type not in self._templates:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self._templates.keys())}")
        return self._templates[agent_type]
    
    def get_system_prompt(self, agent_type: str) -> str:
        """Get system prompt for a specific agent type."""
        template = self.get_template(agent_type)
        return template.SYSTEM_PROMPT
    
    def list_available_templates(self) -> list:
        """List all available prompt templates."""
        return list(self._templates.keys())


# Global prompt manager instance
prompt_manager = PromptManager()