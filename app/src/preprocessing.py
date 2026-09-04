"""Text normalisation for CFPB complaint narratives.

This function is a VERBATIM copy of the one used to fit the models in
notebook 03. If it changes here and not there, serve-time input stops
matching train-time input and accuracy degrades silently.

Deliberately dependency-free (stdlib `re` only) so it cannot drift.
"""
import re


def clean_text(s):
    """Normalise a CFPB complaint narrative. Pure-Python, no dependencies."""
    s = str(s).lower()
    s = re.sub(r"x{2,}[/\-]x{2,}[/\-]x{2,}", " ", s)   # XX/XX/XXXX redacted dates
    s = re.sub(r"\bx{2,}\b", " ", s)                    # XXXX redacted PII
    s = re.sub(r"\{\$[^}]*\}", " ", s)                  # {$1,234.00} redacted amounts
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
