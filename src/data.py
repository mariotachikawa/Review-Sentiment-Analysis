import re

def preprocess_text(text: str) -> str:
    """
    Cleans text by handling quotes, user mentions, URLs, and whitespace.
    """
    text = str(text)
    text = re.sub(r"\\'", "'", text).replace("`", "'")
    text = text.lower()
    text = re.sub(r'@\w+', '@user', text)           # masked user
    text = re.sub(r'http\S+|www\S+', 'http', text)  # masked URL
    text = re.sub(r'\s+', ' ', text).strip()
    return text