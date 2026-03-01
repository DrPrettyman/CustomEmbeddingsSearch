"""Inspect extracted CDP text to understand the actual Q&A format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "cdp" / "apple_2024_cdp.pdf"
doc = fitz.open(str(pdf_path))

# Look at pages 3-8 to find the actual Q&A structure (skip title pages)
for page_num in range(2, min(10, len(doc))):
    text = doc[page_num].get_text()
    print(f"\n{'='*60}")
    print(f"PAGE {page_num + 1}")
    print(f"{'='*60}")
    print(text[:2000])

doc.close()
