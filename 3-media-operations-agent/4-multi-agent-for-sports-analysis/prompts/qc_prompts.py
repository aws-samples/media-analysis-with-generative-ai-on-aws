"""
QC Agent Prompt Templates
"""

class QCPrompts:
    """Prompts for quality control and validation."""
    
    SYSTEM_PROMPT = """You are a quality control assistant that evaluates metadata against requirements and schema format. Use available tools when you need to count words, characters, hashtags, or mentions. Evaluate the metadata systematically and report any violations. If all requirements are met, respond with "Pass"."""
    
    EVALUATION_PROMPT_TEMPLATE = """Evaluate the following metadata against the provided requirements and schema:

METADATA TO EVALUATE:
{metadata}

REQUIREMENTS:
{requirements}

EXPECTED SCHEMA FORMAT:
{schema}

CRITICAL EVALUATION STEPS:
1. Check each requirement systematically
2. If a schema is provided, verify the metadata structure matches EXACTLY
3. Use appropriate tools for quantitative checks (word count, character count, etc.)
4. Report any violations specifically

SCHEMA COMPLIANCE CHECK:
- The metadata structure must match the provided schema exactly
- Check field names, data types, and overall JSON structure
- If the schema shows a specific format, the metadata must follow it precisely
- Schema violations are critical failures

Report "Pass" only if ALL requirements are met AND the schema format is followed exactly. Otherwise, list specific violations."""
    
    @classmethod
    def get_evaluation_prompt(cls, metadata: str, requirements: str, schema: str = "") -> str:
        """Generate comprehensive evaluation prompt."""
        return cls.EVALUATION_PROMPT_TEMPLATE.format(
            metadata=metadata,
            requirements=requirements,
            schema=schema
        )