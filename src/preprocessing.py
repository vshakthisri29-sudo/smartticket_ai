import re

def clean_text(text: str) -> str:
    """
    Clean and normalize customer support ticket text.
    
    Operations:
    - Handles null/missing values gracefully
    - Converts text to lowercase
    - Preserves vital negation words (not, never, no, without)
    - Preserves financial and urgency tokens
    - Strips noisy special characters while preserving sentence/word structure
    - Normalizes multiple whitespaces into a single space
    """
    if text is None:
        return ""
    
    text = str(text).lower().strip()
    if not text:
        return ""
    
    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    
    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove special characters but keep alphanumeric, apostrophes for contractions (don't, can't), and spaces
    text = re.sub(r"[^\w\s\']", " ", text)
    
    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
