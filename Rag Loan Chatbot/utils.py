import re

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text