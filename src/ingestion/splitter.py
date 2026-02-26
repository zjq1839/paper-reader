import re
from typing import List, Dict, Any

class MarkdownSplitter:
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def split(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Split markdown content by H1 headers. 
        Returns a list of sections: {'title': str, 'content': str, 'level': int}
        """
        lines = markdown_content.split('\n')
        sections = []
        current_section = {"title": "Introduction", "content": [], "level": 1}
        
        for line in lines:
            # Simple H1 detection
            if line.startswith('# '):
                # Save previous section
                if current_section["content"]:
                    sections.append({
                        "title": current_section["title"],
                        "content": "\n".join(current_section["content"]).strip(),
                        "level": current_section["level"]
                    })
                
                # Start new section
                current_section = {
                    "title": line.strip('# ').strip(),
                    "content": [],
                    "level": 1
                }
            # Simple H2 detection (optional, for finer grain)
            elif line.startswith('## '):
                 # Save previous section if it's getting long or just treat as subsection
                 # For MVP, we treat H2 as part of the current H1 unless we implement recursive splitting
                 current_section["content"].append(line)
            else:
                current_section["content"].append(line)
        
        # Append last section
        if current_section["content"]:
             sections.append({
                "title": current_section["title"],
                "content": "\n".join(current_section["content"]).strip(),
                "level": current_section["level"]
            })
            
        return sections

    def clean_references(self, content: str) -> str:
        """
        Remove References section if present.
        """
        # Simple heuristic: stop at a line that is just "References" or "# References"
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            if re.match(r'^#*\s*References\s*$', line, re.IGNORECASE):
                break
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
