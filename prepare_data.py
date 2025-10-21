import os
import json
import re
import argparse
from typing import Iterator, List, Dict, Any, Optional


# =============================
# Configuration
# =============================

# Input file path
INPUT_PATH = os.environ.get("INPUT_PATH", "dataset.csv")

# Grouping: "http" (default, start by http/https) or "fixed" (N lines/article)
GROUP_MODE = os.environ.get("GROUP_MODE", "http").lower()

# Only used when GROUP_MODE=fixed
LINES_PER_ARTICLE = int(os.environ.get("LINES_PER_ARTICLE", "14"))

# Vietnamese word segmenter
SEGMENTER = os.environ.get("SEGMENTER", "custom").lower()  # custom | pyvi | underthesea | vncore

# Output files (JSONL)
ARTICLES_JSONL = os.environ.get("ARTICLES_JSONL", "articles.jsonl")
SEGMENTS_JSONL = os.environ.get("SEGMENTS_JSONL", "segments.jsonl")
SENTENCES_JSONL = os.environ.get("SENTENCES_JSONL", "sentences.jsonl")


# =============================
# I/O utilities
# =============================

def read_lines_stream(path: str, encoding_candidates: Optional[List[str]] = None) -> Iterator[str]:
    """Stream lines with multiple encodings; ignore decode errors."""
    if encoding_candidates is None:
        encoding_candidates = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    last_error: Optional[Exception] = None
    for enc in encoding_candidates:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for ln in f:
                    yield ln.rstrip("\n")
            return
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Cannot read {path}; last error: {last_error}")


def iter_articles_fixed_blocks(path: str, lines_per_article: int) -> Iterator[Dict[str, Any]]:
    """Yield one article per N lines; join lines into 'content'."""
    buffer: List[str] = []
    article_index = 0
    for ln in read_lines_stream(path):
        buffer.append(ln)
        if len(buffer) >= lines_per_article:
            content = "\n".join(buffer).strip()
            yield {
                "doc_id": f"doc-{article_index:06d}",
                "content": content,
            }
            buffer = []
            article_index += 1
    # Flush remainder
    if buffer:
        content = "\n".join(buffer).strip()
        yield {
            "doc_id": f"doc-{article_index:06d}",
            "content": content,
        }


HTTP_PREFIXES = ("http://", "https://")

def iter_articles_http_start(path: str) -> Iterator[Dict[str, Any]]:
    """Start new article at lines beginning with http/https; first line has 8 CSV fields."""
    header_skipped = False
    buffer: List[str] = []
    article_index = 0

    for ln in read_lines_stream(path):
        if not header_skipped:
            if ln.lower().startswith("original_link,title,domain,category,keywords,description,publication_date,content"):
                header_skipped = True
                continue
            # If there is no header, proceed directly
            header_skipped = True

        if ln.startswith(HTTP_PREFIXES):
            # flush previous
            if buffer:
                rec = _build_record_from_block(buffer, article_index)
                yield rec
                article_index += 1
                buffer = []
            buffer.append(ln)
        else:
            # continuation of current article block
            buffer.append(ln)

    # flush last
    if buffer:
        rec = _build_record_from_block(buffer, article_index)
        yield rec


def _build_record_from_block(block_lines: List[str], article_index: int) -> Dict[str, Any]:
    """Parse first line into 8 fields (max 7 commas); rest is content."""
    first = block_lines[0] if block_lines else ""
    rest_lines = block_lines[1:] if len(block_lines) > 1 else []

    parts = first.split(",", 7)
    # Ensure we have 8 fields; pad with empty strings if necessary
    while len(parts) < 8:
        parts.append("")

    original_link = parts[0].strip()
    title = parts[1].strip()
    domain = parts[2].strip()
    category = parts[3].strip()
    keywords = parts[4].strip()
    description = parts[5].strip()
    publication_date = parts[6].strip()
    content_start = parts[7]

    content_full = "\n".join([content_start] + rest_lines).strip()

    # Strip surrounding quotes if present
    def dequote(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1].strip()
        return s

    rec = {
        "doc_id": f"doc-{article_index:06d}",
        "original_link": dequote(original_link),
        "title": dequote(title),
        "domain": dequote(domain),
        "category": dequote(category),
        "keywords": dequote(keywords),
        "description": dequote(description),
        "publication_date": dequote(publication_date),
        "content": content_full,
    }
    return rec


def write_jsonl(path: str, records: Iterator[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# =============================
# Paragraph/Sentence split
# =============================

PARA_SPLIT_RE = re.compile(r"\n{2,}")

def segment_paragraphs(text: str) -> List[Dict[str, Any]]:
    """Split paragraphs on blank lines; keep char offsets."""
    paragraphs: List[Dict[str, Any]] = []
    start = 0
    for match in PARA_SPLIT_RE.finditer(text):
        end = match.start()
        chunk = text[start:end].strip()
        if chunk:
            paragraphs.append({"text": chunk, "start_char": start, "end_char": end})
        start = match.end()
    # tail
    tail = text[start:].strip()
    if tail:
        paragraphs.append({"text": tail, "start_char": start, "end_char": start + len(tail)})
    return paragraphs


# Simple Vietnamese-aware sentence splitter (heuristic)
SENT_BOUNDARY_RE = re.compile(r"([.!?…]+)(\s+)")
ABBREVIATIONS = {
    "TP.", "TP.HCM", "TS.", "ThS.", "PGS.", "GS.", "Mr.", "Mrs.", "Dr.",
}

def split_sentences(text: str, para_start_offset: int) -> List[Dict[str, Any]]:
    sentences: List[Dict[str, Any]] = []
    idx = 0
    acc = []
    for m in SENT_BOUNDARY_RE.finditer(text):
        end_idx = m.end(1)  # end of punctuation
        piece = text[idx:end_idx]
        acc.append(piece)
        candidate = "".join(acc).strip()
        acc = []
        # If ends with abbreviation, don't split
        tail = candidate.split()[-1] if candidate.split() else ""
        if tail in ABBREVIATIONS:
            acc.append(candidate + m.group(2))
            idx = m.end()
            continue
        start_char = para_start_offset + (idx)
        end_char = para_start_offset + end_idx
        sentences.append({
            "text": candidate,
            "start_char": start_char,
            "end_char": end_char,
        })
        idx = m.end()
    # Remainder
    if idx < len(text):
        rest = text[idx:].strip()
        if rest:
            sentences.append({
                "text": rest,
                "start_char": para_start_offset + idx,
                "end_char": para_start_offset + len(text),
            })
    return sentences


# =============================
# Vietnamese word segmentation
# =============================

_segmenter_cached = None

def _load_segmenter():
    global _segmenter_cached
    if _segmenter_cached is not None:
        return _segmenter_cached
    seg_name = SEGMENTER
    try:
        if seg_name == "custom":
            from custom_segmenter import CustomVietnameseSegmenter
            segmenter = CustomVietnameseSegmenter()
            _segmenter_cached = (seg_name, segmenter.segment)
        elif seg_name == "pyvi":
            from pyvi import ViTokenizer  # type: ignore
            _segmenter_cached = (seg_name, ViTokenizer.tokenize)
        elif seg_name == "underthesea":
            from underthesea import word_tokenize  # type: ignore
            _segmenter_cached = (seg_name, lambda s: " ".join(word_tokenize(s)))
        elif seg_name == "vncore":
            # Requires vncorenlp and model .jar
            from vncorenlp import VnCoreNLP  # type: ignore
            rdr = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators=["wseg"], max_heap_size='-Xmx2g')
            def vncore_tokenize(s: str) -> str:
                # Returns list of sentences; each is list of tokens
                sents = rdr.word_segment(s)
                tokens = []
                for sen in sents:
                    tokens.extend(sen)
                return " ".join(tokens)
            _segmenter_cached = (seg_name, vncore_tokenize)
        elif seg_name == "underthesea_advanced":
            # Underthesea + normalization
            from underthesea import word_tokenize, text_normalize  # type: ignore
            def advanced_tokenize(s: str) -> str:
                normalized = text_normalize(s)
                tokens = word_tokenize(normalized)
                return " ".join(tokens)
            _segmenter_cached = (seg_name, advanced_tokenize)
        elif seg_name == "pyvi_advanced":
            # PyVi + POS tagging
            from pyvi import ViTokenizer, ViPosTagger  # type: ignore
            def advanced_pyvi_tokenize(s: str) -> str:
                tokens, tags = ViPosTagger.postagging(ViTokenizer.tokenize(s))
                return " ".join(tokens)
            _segmenter_cached = (seg_name, advanced_pyvi_tokenize)
        elif seg_name == "underthesea_ws":
            # Underthesea: sentence then word tokenize
            from underthesea import word_tokenize, text_normalize, sent_tokenize  # type: ignore
            def ws_tokenize(s: str) -> str:
                sentences = sent_tokenize(s)
                all_tokens = []
                for sent in sentences:
                    normalized = text_normalize(sent)
                    tokens = word_tokenize(normalized)
                    all_tokens.extend(tokens)
                return " ".join(all_tokens)
            _segmenter_cached = (seg_name, ws_tokenize)
        else:
            _segmenter_cached = ("noop", lambda s: s)
    except Exception as e:
        print(f"Warning: Failed to load segmenter '{seg_name}': {e}")
        _segmenter_cached = ("noop", lambda s: s)
    return _segmenter_cached

def segment_words(text: str) -> str:
    name, fn = _load_segmenter()
    segmented = fn(text)
    # Normalize: collapse extra spaces
    segmented = re.sub(r"\s+", " ", segmented).strip()
    return segmented


# =============================
# Pipeline
# =============================

def step_1_convert_to_jsonl(input_path: str, out_path: str, lines_per_article: int, group_mode: str) -> int:
    """Step 1: Convert raw file to JSONL with stable content field.

    Modes:
    - http (default): each article starts at a line beginning with http/https.
    - fixed: group by N lines per article.
    """
    if group_mode == "fixed":
        return write_jsonl(out_path, iter_articles_fixed_blocks(input_path, lines_per_article))
    # default: http mode
    return write_jsonl(out_path, iter_articles_http_start(input_path))


def step_2_segment_articles(articles_path: str, out_segments_path: str) -> int:
    """Step 2: Word-segment entire articles (no paragraph splitting)."""
    count = 0
    with open(articles_path, "r", encoding="utf-8") as f_in, open(out_segments_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "")
            content = rec.get("content", "")
            # Word-segment the entire article content
            content_ws = segment_words(content)
            out = {
                "doc_id": doc_id,
                "content_ws": content_ws,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def step_3_sentence_split(segments_path: str, out_sentences_path: str) -> int:
    """Step 3: Split word-segmented articles into sentences."""
    count = 0
    with open(segments_path, "r", encoding="utf-8") as f_in, open(out_sentences_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "")
            # Use word-segmented article content for sentence splitting
            text = rec.get("content_ws", "")
            sents = split_sentences(text, 0)  # start_offset = 0 since we're working with full article
            for sid, s in enumerate(sents):
                out = {
                    "doc_id": doc_id,
                    "sent_id": f"{doc_id}-s{sid}",
                    "sentence_index": sid,
                    "sentence": s["text"],  # This is word-segmented sentence
                    "start_char": s["start_char"],
                    "end_char": s["end_char"],
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    global INPUT_PATH, GROUP_MODE, LINES_PER_ARTICLE
    global ARTICLES_JSONL, SEGMENTS_JSONL, SENTENCES_JSONL
    global SEGMENTER

    parser = argparse.ArgumentParser(description="Prepare Vietnamese news data: articles -> segments -> sentences")
    parser.add_argument("--input-path", default=INPUT_PATH, help="Path to raw file (default: sample.csv)")
    parser.add_argument("--group-mode", choices=["http", "fixed"], default=GROUP_MODE, help="Grouping mode: http (start with http/https) or fixed")
    parser.add_argument("--lines-per-article", type=int, default=LINES_PER_ARTICLE, help="Used only in fixed mode")
    parser.add_argument("--articles-jsonl", default=ARTICLES_JSONL, help="Output path for articles.jsonl")
    parser.add_argument("--segments-jsonl", default=SEGMENTS_JSONL, help="Output path for segments.jsonl")
    parser.add_argument("--sentences-jsonl", default=SENTENCES_JSONL, help="Output path for sentences.jsonl")
    parser.add_argument("--segmenter", choices=["custom", "pyvi", "underthesea", "vncore", "underthesea_advanced", "pyvi_advanced", "underthesea_ws"], default=SEGMENTER, help="Segmenter backend to use")

    args = parser.parse_args()

    # Override globals from CLI for downstream helpers
    INPUT_PATH = args.input_path
    GROUP_MODE = args.group_mode
    LINES_PER_ARTICLE = args.lines_per_article
    ARTICLES_JSONL = args.articles_jsonl
    SEGMENTS_JSONL = args.segments_jsonl
    SENTENCES_JSONL = args.sentences_jsonl
    SEGMENTER = args.segmenter

    if GROUP_MODE == "fixed":
        print(f"[Step 1] Converting raw -> {ARTICLES_JSONL} using GROUP_MODE=fixed, LINES_PER_ARTICLE={LINES_PER_ARTICLE}")
    else:
        print(f"[Step 1] Converting raw -> {ARTICLES_JSONL} using GROUP_MODE=http (split by http/https)")
    n_articles = step_1_convert_to_jsonl(INPUT_PATH, ARTICLES_JSONL, LINES_PER_ARTICLE, GROUP_MODE)
    print(f"  Wrote {n_articles} articles")

    # print(f"[Step 2] Segmenting articles with word segmentation -> {SEGMENTS_JSONL} (SEGMENTER={SEGMENTER})")
    # n_segments = step_2_segment_articles(ARTICLES_JSONL, SEGMENTS_JSONL)
    # print(f"  Wrote {n_segments} segments")

    # print(f"[Step 3] Splitting word-segmented sentences -> {SENTENCES_JSONL}")
    # n_sents = step_3_sentence_split(SEGMENTS_JSONL, SENTENCES_JSONL)
    # print(f"  Wrote {n_sents} sentences")


if __name__ == "__main__":
    main()


