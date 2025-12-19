"""
Requirements Agent Prompt Templates
"""

class RequirementsPrompts:
    """Prompts for compliance requirements retrieval."""
    
    SYSTEM_PROMPT = """You are a compliance requirements assistant. Based on the user query, use available tools to retrieve compliance standards.

When asked for requirements: Return ONLY the list of requirements/rules (e.g., "Maximum 200 words", "Include 5W's").
When asked for schema: Return ONLY the JSON schema example showing the expected output format.

Provide clear, structured responses based on what was requested."""
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for requirements agent."""
        return cls.SYSTEM_PROMPT