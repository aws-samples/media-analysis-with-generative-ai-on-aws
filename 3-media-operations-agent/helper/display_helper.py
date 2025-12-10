"""
Display helper functions for Jupyter notebooks.
Provides formatted output for success messages, info, and results.
"""

from IPython.display import display, HTML, Markdown


def print_success(message: str):
    """Display a success message with green styling."""
    display(HTML(
        f'<div style="padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; '
        f'border-radius: 5px; color: #155724;">'
        f'<b>✅ SUCCESS:</b> {message}</div>'
    ))


def print_info(message: str):
    """Display an info message with blue styling."""
    display(HTML(
        f'<div style="padding: 10px; background-color: #d1ecf1; border: 1px solid #bee5eb; '
        f'border-radius: 5px; color: #0c5460;">'
        f'<b>ℹ️ INFO:</b> {message}</div>'
    ))


def print_warning(message: str):
    """Display a warning message with yellow styling."""
    display(HTML(
        f'<div style="padding: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; '
        f'border-radius: 5px; color: #856404;">'
        f'<b>⚠️ WARNING:</b> {message}</div>'
    ))


def print_error(message: str):
    """Display an error message with red styling."""
    display(HTML(
        f'<div style="padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; '
        f'border-radius: 5px; color: #721c24;">'
        f'<b>❌ ERROR:</b> {message}</div>'
    ))


def print_result(title: str, content: str):
    """Display a formatted result box with title and content."""
    # Convert literal \n to actual newlines
    content = content.replace('\\n', '\n')
    display(HTML(f'''
        <div style="padding: 15px; background-color: #e7f3ff; border-left: 4px solid #2196F3; margin: 10px 0;">
            <h4 style="margin-top: 0; color: #000000;">{title}</h4>
            <pre style="background-color: white; padding: 10px; border-radius: 3px; color: #000000; 
                        white-space: pre-wrap; font-family: inherit; margin: 0;">{content}</pre>
        </div>
    '''))


def print_code(code: str, language: str = "python"):
    """Display formatted code block."""
    display(Markdown(f"```{language}\n{code}\n```"))


def print_section_header(title: str, description: str = None):
    """Display a formatted section header."""
    html = f'<div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); ' \
           f'border-radius: 10px; color: white; margin: 20px 0;">' \
           f'<h2 style="margin: 0; color: white;">{title}</h2>'
    
    if description:
        html += f'<p style="margin: 10px 0 0 0; font-size: 16px;">{description}</p>'
    
    html += '</div>'
    display(HTML(html))
