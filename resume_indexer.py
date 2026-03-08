"""
resume_indexer.py
-----------------
Production-ready PDF indexer for EnhanceCV two-column resumes.
Outputs clean chunks ready for FAISS + MiniLM-L6-v2 (your existing RAG setup).

Usage:
    python resume_indexer.py --pdf MohitKumarResume.pdf --out index_data.json
"""

import re
import json
import argparse
import pdfplumber

# ─────────────────────────────────────────────────────────────────
# STEP 1: Extract — two-column aware
# ─────────────────────────────────────────────────────────────────

def extract_two_column(pdf_path: str, col_split: float = 0.52) -> str:
    """
    EnhanceCV resumes use a ~52% left / 48% right two-column layout.
    We crop each column separately so content never gets interleaved,
    then concatenate left + right as one clean string.
    """
    left_pages, right_pages = [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            mid = page.width * col_split

            left_text  = page.crop((0,   0, mid,         page.height)).extract_text() or ""
            right_text = page.crop((mid, 0, page.width,  page.height)).extract_text() or ""

            left_pages.append(left_text.strip())
            right_pages.append(right_text.strip())

    # Left column contains the main timeline (Summary, Experience, Education)
    # Right column contains Key Achievements, Projects, Skills
    full_text = "\n\n".join(left_pages) + "\n\n" + "\n\n".join(right_pages)
    return full_text


# ─────────────────────────────────────────────────────────────────
# STEP 2: Clean — fix known EnhanceCV artifacts
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove PDF extraction noise specific to EnhanceCV exports."""

    # Remove zero-width / invisible Unicode characters (​ \u200b, \u0000, etc.)
    text = re.sub(r'[\u0000\u200b\u200c\u200d\ufeff\u00ad]', '', text)

    # Fix broken "+" signs in phone numbers (\x00 91 → +91)
    text = re.sub(r'\s*\x00\s*(\d)', r'+\1', text)

    # Normalize "4.5+" years — the + sometimes comes through as \u0000
    text = re.sub(r'(\d)[\u0000]+', r'\1+', text)

    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


# ─────────────────────────────────────────────────────────────────
# STEP 3: Chunk — section-aware splitting for better RAG retrieval
# ─────────────────────────────────────────────────────────────────

SECTION_HEADERS = [
    "SUMMARY", "EXPERIENCE", "EDUCATION", "KEY ACHIEVEMENTS",
    "INDEPENDENT PROJECTS", "SKILLS", "CERTIFICATIONS", "PUBLICATIONS"
]

def chunk_by_section(text: str) -> list[dict]:
    """
    Split text at known resume section headers.
    Each chunk = {section, content, char_count}.
    Falls back to fixed-size chunking for very long sections.
    """
    # Build a regex that detects any section header on its own line
    header_pattern = re.compile(
        r'^(' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')$',
        re.MULTILINE | re.IGNORECASE
    )

    chunks = []
    positions = [(m.start(), m.group()) for m in header_pattern.finditer(text)]

    if not positions:
        # No headers found — fall back to plain paragraph splitting
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                chunks.append({"section": "general", "content": para, "char_count": len(para)})
        return chunks

    # Add a sentinel at the end
    positions.append((len(text), "END"))

    for i, (start, header) in enumerate(positions[:-1]):
        end = positions[i + 1][0]
        content = text[start:end].strip()

        if len(content) <= 1200:
            # Small enough — keep as one chunk
            chunks.append({
                "section":    header,
                "content":    content,
                "char_count": len(content)
            })
        else:
            # Large section — split into ~600 char sub-chunks with 100 char overlap
            sub_chunks = sliding_window(content, size=600, overlap=100)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "section":    f"{header} (part {j+1})",
                    "content":    sub,
                    "char_count": len(sub)
                })

    return chunks


def sliding_window(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Fixed-size chunking with overlap, respecting word boundaries."""
    words = text.split()
    chunks, start = [], 0
    # Approximate words per chunk
    words_per_chunk = size // 5  # ~5 chars/word average
    step = words_per_chunk - (overlap // 5)

    while start < len(words):
        chunk_words = words[start: start + words_per_chunk]
        chunks.append(" ".join(chunk_words))
        start += step

    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────
# STEP 4: Validate — surface any remaining problems
# ─────────────────────────────────────────────────────────────────

EXPECTED_CONTENT = [
    ("Name",        "MOHIT KUMAR"),
    ("Email",       "mohitkumartanwar"),
    ("Current role","Data Scientist"),
    ("Employer",    "XPO"),
    ("LLM work",    "Text2SQL"),
    ("Education",   "Indian Institute"),
    ("Skill: GCP",  "GCP"),
    ("Project",     "RAG"),
    ("Achievement", "120,000"),
]

def validate(text: str):
    print("\n── Validation ──────────────────────────────────")
    all_pass = True
    for label, term in EXPECTED_CONTENT:
        found = term.lower() in text.lower()
        status = "✅" if found else "❌ MISSING"
        if not found:
            all_pass = False
        print(f"  {status}  {label}: '{term}'")

    junk_chars = re.findall(r'[\u0000\u200b\ufeff]', text)
    if junk_chars:
        print(f"\n  ⚠️  {len(junk_chars)} invisible/junk characters remain after cleaning")
        all_pass = False
    else:
        print(f"\n  ✅  No invisible/junk characters detected")

    print(f"\n  {'✅ All checks passed' if all_pass else '⚠️  Some checks failed — review output'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run(pdf_path: str, out_path: str = "index_data.json", col_split: float = 0.52):
    print(f"📄 Input:  {pdf_path}")
    print(f"📦 Output: {out_path}\n")

    # 1. Extract
    print("── Step 1: Extracting (two-column aware) ───────")
    raw = extract_two_column(pdf_path, col_split)
    print(f"  Extracted {len(raw)} characters from PDF")

    # 2. Clean
    print("── Step 2: Cleaning ────────────────────────────")
    clean = clean_text(raw)
    removed = len(raw) - len(clean)
    print(f"  Removed {removed} noise characters")

    # 3. Chunk
    print("── Step 3: Chunking by section ─────────────────")
    chunks = chunk_by_section(clean)
    print(f"  Created {len(chunks)} chunks")
    for c in chunks:
        print(f"    [{c['section'][:30]:<30}] {c['char_count']} chars")

    # 4. Validate
    validate(clean)

    # 5. Save
    output = {
        "source":     pdf_path,
        "full_text":  clean,
        "chunk_count": len(chunks),
        "chunks":     chunks
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done — saved to {out_path}")
    print("\n── Sample cleaned full text (first 600 chars) ──")
    print(clean[:600])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",       default="me/MohitKumarResume.pdf")
    parser.add_argument("--out",       default="me/index_data.json")
    parser.add_argument("--col_split", type=float, default=0.52,
                        help="Column split ratio (0.52 = 52%% left column)")
    args = parser.parse_args()
    run(args.pdf, args.out, args.col_split)