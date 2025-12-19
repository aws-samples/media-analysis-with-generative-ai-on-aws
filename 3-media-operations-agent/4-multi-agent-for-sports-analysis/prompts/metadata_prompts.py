"""
Metadata Agent Prompt Templates
"""

class MetadataPrompts:
    """Prompts for metadata generation and enhancement."""
    
    CREATE_SYSTEM_PROMPT = """You are a metadata generation assistant. Analyze the provided video information and generate compliant metadata following the requirements and output format."""
    
    ENHANCE_SYSTEM_PROMPT = """You are a metadata enhancement assistant. Improve the existing metadata based on feedback, using the video analysis and requirements as reference."""
    
    MAIN_SYSTEM_PROMPT = """You are a metadata assistant that generates or enhances video metadata. Use available tools to create or improve metadata based on the provided information. Always base your output on factual information provided, and follow the specified requirements and format."""
    
    CREATE_PROMPT_TEMPLATE = """Here is the video analysis: {video_analysis}
Here is additional info about the video: {additional_info}
Here is the requirements: {requirements}
Please generate and output the metadata in following output format ONLY.
{output_format}"""
    
    ENHANCE_PROMPT_TEMPLATE = """Here is the current metadata: {current_metadata}
Here is the feedback: {feedback}
Here is the video analysis: {video_analysis}
Here is additional info about the video: {additional_info}
Here is the requirements: {requirements}
Please generate and output the metadata in following output format ONLY.
{output_format}"""
    
    MAIN_AGENT_CREATE_PROMPT_TEMPLATE = """Here is the video analysis: {video_analysis}
Here is additional info about the video: {additional_info}
Here is the requirements: {requirements}
Here is the output format: {output_format}"""
    
    MAIN_AGENT_ENHANCE_PROMPT_TEMPLATE = """Here is the current metadata: {current_metadata}
Here is the feedback: {feedback}
Here is the video analysis: {video_analysis}
Here is additional info about the video: {additional_info}
Here is the requirements: {requirements}
Here is the output format: {output_format}"""
    
    @classmethod
    def get_create_prompt(cls, video_analysis: str, additional_info: str, 
                         requirements: str, output_format: str) -> str:
        """Generate creation prompt with parameters."""
        return cls.CREATE_PROMPT_TEMPLATE.format(
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format
        )
    
    @classmethod
    def get_enhance_prompt(cls, current_metadata: str, feedback: str, 
                          video_analysis: str, additional_info: str,
                          requirements: str, output_format: str) -> str:
        """Generate enhancement prompt with parameters."""
        return cls.ENHANCE_PROMPT_TEMPLATE.format(
            current_metadata=current_metadata,
            feedback=feedback,
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format
        )
    
    @classmethod
    def get_main_agent_create_prompt(cls, video_analysis: str, additional_info: str,
                                   requirements: str, output_format: str) -> str:
        """Generate main agent prompt for creation mode."""
        return cls.MAIN_AGENT_CREATE_PROMPT_TEMPLATE.format(
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format
        )
    
    @classmethod
    def get_main_agent_enhance_prompt(cls, current_metadata: str, feedback: str,
                                    video_analysis: str, additional_info: str,
                                    requirements: str, output_format: str) -> str:
        """Generate main agent prompt for enhancement mode."""
        return cls.MAIN_AGENT_ENHANCE_PROMPT_TEMPLATE.format(
            current_metadata=current_metadata,
            feedback=feedback,
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format
        )