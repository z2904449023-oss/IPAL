# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm

# PDF extraction dependency (PyMuPDF)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# =============================================================================
# 0) Shared utilities
# =============================================================================

TAG_RE = re.compile(r"<[^>]+>")

def normalize_text_basic(s: str) -> str:
    """Keep newlines; normalize BOM/nbsp; collapse spaces; standardize line breaks."""
    s = s or ""
    s = s.replace("\ufeff", "")
    s = s.replace("\u00a0", " ")
    s = s.replace("\u200b", "")
    s = html.unescape(s)
    s = TAG_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha1_text(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()

def clamp_int(x, lo: int, hi: int) -> int:
    try:
        x = int(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))

def compute_score(dir_: int, intensity: int, certainty: int) -> float:
    if dir_ == 0:
        return 0.0
    score = 0.5 * (intensity / 3.0) + 0.5 * (certainty / 2.0)
    return max(0.0, min(1.0, float(score)))

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

def list_files(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(p)
    out.sort()
    return out


# =============================================================================
# 1) PDF -> CSV extractor (TOC-based, improved)
# =============================================================================

DEFAULT_TOC_SCAN_PAGES = 15
DEFAULT_TOC_DOT_THRESHOLD = 6
DEFAULT_MIN_TITLE_LEN = 4
DEFAULT_MAX_TITLE_LEN = 220
DEFAULT_MAX_PAGE_GAP = 80  # set -1 to disable via CLI

def is_bad_title(title: str, min_len: int, max_len: int) -> bool:
    if not title:
        return True
    t = normalize_text_basic(title)
    if len(t) < min_len or len(t) > max_len:
        return True
    if re.fullmatch(r"[\W_]+", t):  # all symbols
        return True
    bad_prefix = ("page ", "©", "copyright", "文件", "to subscribe to")
    if t.lower().startswith(bad_prefix):
        return True
    return False

def clean_body_factiva(text: str) -> str:
    """Remove common headers/footers but keep Factiva meta lines like date/source."""
    if not text:
        return ""
    text = normalize_text_basic(text)

    # Remove "Page x of y"
    text = re.sub(r"(?im)^Page\s+\d+\s+of\s+\d+.*?$", "", text)

    # Remove some copyright/subscription boilerplate
    text = re.sub(r"(?im)^©\s*\d{4}.*?$", "", text)
    text = re.sub(r"(?im)^Copyright\s+©?\s*\d{4}.*?$", "", text)
    text = re.sub(r"(?im)^To subscribe to Barron's,.*?$", "", text)

    # Remove file code line in Chinese if it appears as a header
    text = re.sub(r"(?im)^文件\s+[A-Z0-9]+.*?$", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

TOC_LINE_RE = re.compile(
    r"""^(?P<title>.+?)
         (?:\.{3,}|\s{2,})\s*
         (?P<page>\d{1,4})\s*$""",
    re.VERBOSE
)
TOC_PAGEONLY_RE = re.compile(r"^\.*\s*(\d{1,4})\s*$")

def detect_toc_pages(pages_text: List[str], toc_scan_pages: int, dot_threshold: int) -> List[int]:
    toc_pages = []
    for i, text in enumerate(pages_text[:max(1, toc_scan_pages)]):
        t = normalize_text_basic(text)
        if len(re.findall(r"\.{3,}\s*\d{1,4}\b", t)) >= dot_threshold:
            toc_pages.append(i)
    return toc_pages

def parse_toc_entries(toc_text: str, total_pages: int, min_title_len: int, max_title_len: int) -> List[Tuple[str, int]]:
    lines = [ln.rstrip() for ln in toc_text.splitlines()]
    entries: List[Tuple[str, int]] = []
    buf = ""

    for raw in lines:
        ln = normalize_text_basic(raw)
        if not ln:
            continue

        if re.match(r"(?i)^page\s+\d+\s+of\s+\d+", ln) or ln.startswith("©"):
            continue

        m = TOC_LINE_RE.match(ln)
        if m:
            title_part = normalize_text_basic(m.group("title"))
            page = int(m.group("page"))

            title = normalize_text_basic((buf + " " + title_part).strip())
            buf = ""

            if 1 <= page <= total_pages and not is_bad_title(title, min_title_len, max_title_len):
                entries.append((title, page))
            continue

        m2 = TOC_PAGEONLY_RE.match(ln)
        if m2 and buf:
            page = int(m2.group(1))
            title = normalize_text_basic(buf)
            buf = ""
            if 1 <= page <= total_pages and not is_bad_title(title, min_title_len, max_title_len):
                entries.append((title, page))
            continue

        buf = normalize_text_basic((buf + " " + ln).strip()) if buf else ln

    seen = set()
    uniq: List[Tuple[str, int]] = []
    for t, p in entries:
        key = (t, p)
        if key not in seen:
            seen.add(key)
            uniq.append((t, p))
    return uniq

def infer_title_from_body(body: str) -> str:
    if not body:
        return ""
    lines = [normalize_text_basic(x) for x in body.splitlines() if normalize_text_basic(x)]
    if not lines:
        return ""

    skip_exact = {"MARKET WEEK", "Personal Finance"}
    skip_prefix = ("By ", "Copyright", "(FROM ", "(END)", "January ", "文件 ", "Page ")

    cand = []
    for ln in lines[:14]:
        if ln in skip_exact:
            continue
        if any(ln.startswith(p) for p in skip_prefix):
            continue
        # "169 字" / "1,481 字" / "169 words"
        if re.search(r"(?i)\b\d[\d,]*\s*(字|words?)\b", ln):
            continue
        # date-ish lines
        if re.search(r"\d{4}\s*年", ln) or re.search(r"(?i)\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", ln):
            continue
        cand.append(ln)

    if not cand:
        return ""

    cand_sorted = sorted(cand, key=lambda x: len(x), reverse=True)
    title = cand_sorted[0]
    title = re.sub(r"^\*+", "", title).strip()
    return title

def extract_factiva_pdf_to_df(
    pdf_path: str,
    toc_scan_pages: int = DEFAULT_TOC_SCAN_PAGES,
    toc_dot_threshold: int = DEFAULT_TOC_DOT_THRESHOLD,
    min_title_len: int = DEFAULT_MIN_TITLE_LEN,
    max_title_len: int = DEFAULT_MAX_TITLE_LEN,
    max_page_gap: Optional[int] = DEFAULT_MAX_PAGE_GAP,
) -> pd.DataFrame:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages_text = [page.get_text("text") for page in doc]
    total_pages = len(pages_text)

    toc_pages = detect_toc_pages(pages_text, toc_scan_pages=toc_scan_pages, dot_threshold=toc_dot_threshold)
    if not toc_pages:
        raise ValueError(f"[PDF] 未识别到目录页: {pdf_path}")

    toc_text = "\n".join(pages_text[i] for i in toc_pages)
    toc_entries = parse_toc_entries(toc_text, total_pages, min_title_len=min_title_len, max_title_len=max_title_len)
    if not toc_entries:
        raise ValueError(f"[PDF] 目录解析为空: {pdf_path}")

    # stable sort: by page then occurrence
    toc_entries_sorted = sorted(enumerate(toc_entries), key=lambda x: (x[1][1], x[0]))
    toc_entries_sorted = [x[1] for x in toc_entries_sorted]

    items = []
    for i, (toc_title, start_page) in enumerate(toc_entries_sorted):
        end_page = total_pages
        for j in range(i + 1, len(toc_entries_sorted)):
            next_page = toc_entries_sorted[j][1]
            if next_page > start_page:
                end_page = next_page - 1
                break

        if max_page_gap is not None and (end_page - start_page) > max_page_gap:
            end_page = min(start_page + max_page_gap, total_pages)

        raw_content = "\n".join(pages_text[start_page - 1:end_page])
        content = clean_body_factiva(raw_content)

        fixed_title = normalize_text_basic(toc_title)
        if is_bad_title(fixed_title, min_title_len, max_title_len):
            inferred = infer_title_from_body(content)
            if not is_bad_title(inferred, min_title_len, max_title_len):
                fixed_title = inferred

        fixed_title = normalize_text_basic(fixed_title)
        if is_bad_title(fixed_title, min_title_len, max_title_len):
            continue

        items.append({
            "标题": fixed_title,
            "起始页": int(start_page),
            "结束页": int(end_page),
            "正文": content,
            "source_file": os.path.basename(pdf_path),
            "source_path": os.path.abspath(pdf_path),
            "source_type": "pdf",
        })

    return pd.DataFrame(items)


# =============================================================================
# 2) RTF -> CSV extractor (Factiva RTF, heuristic segmentation)
# =============================================================================

RTF_HEX_RE = re.compile(r"\\'([0-9a-fA-F]{2})")
RTF_UNI_RE = re.compile(r"\\u(-?\d+)\??")
RTF_CTRLWORD_RE = re.compile(r"\\[a-zA-Z]+\d* ?")
RTF_BRACES_RE = re.compile(r"[{}]")

def rtf_to_text(rtf: str) -> str:
    if not rtf:
        return ""

    s = rtf
    s = s.replace("\\par", "\n").replace("\\line", "\n")

    def _hex_repl(m):
        try:
            b = bytes([int(m.group(1), 16)])
            return b.decode("latin-1", errors="ignore")
        except Exception:
            return ""
    s = RTF_HEX_RE.sub(_hex_repl, s)

    def _uni_repl(m):
        try:
            code = int(m.group(1))
            if code < 0:
                code = code + 65536
            return chr(code)
        except Exception:
            return ""
    s = RTF_UNI_RE.sub(_uni_repl, s)

    s = RTF_CTRLWORD_RE.sub("", s)
    s = RTF_BRACES_RE.sub("", s)
    s = html.unescape(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def looks_like_factiva_title(line: str, min_len: int, max_len: int) -> bool:
    line = normalize_text_basic(line)
    if is_bad_title(line, min_len, max_len):
        return False
    if re.search(r"(?i)\b\d[\d,]*\s*(字|words?)\b", line):
        return False
    if re.search(r"(?i)^(dow jones|reuters|bloomberg)\b", line) and len(line) < 10:
        return False
    return True

def segment_factiva_articles_from_text(
    text: str,
    min_title_len: int = DEFAULT_MIN_TITLE_LEN,
    max_title_len: int = DEFAULT_MAX_TITLE_LEN,
) -> List[Dict]:
    """
    Heuristic segmentation for Factiva RTF plain text.
    """
    t = normalize_text_basic(text)
    lines = [ln.rstrip() for ln in t.splitlines()]
    norm_lines = [normalize_text_basic(ln) for ln in lines]

    def is_len_line(s: str) -> bool:
        return bool(re.search(r"(?i)^\s*\d[\d,]*\s*(字|words?)\s*$", s))

    starts: List[int] = []
    i = 0
    while i < len(norm_lines):
        a = norm_lines[i]
        b = norm_lines[i + 1] if i + 1 < len(norm_lines) else ""

        if a and b and a == b and looks_like_factiva_title(a, min_title_len, max_title_len):
            starts.append(i)
            i += 2
            continue

        if a and looks_like_factiva_title(a, min_title_len, max_title_len):
            window = norm_lines[i + 1:i + 4]
            if any(is_len_line(x) for x in window):
                starts.append(i)
                i += 1
                continue

        i += 1

    starts = sorted(set(starts))

    if not starts:
        body = t
        title = infer_title_from_body(body) or "UNKNOWN_TITLE"
        return [{
            "标题": title,
            "起始页": -1,
            "结束页": -1,
            "正文": body,
        }]

    items: List[Dict] = []
    for idx, st in enumerate(starts):
        ed = starts[idx + 1] if idx + 1 < len(starts) else len(norm_lines)
        block_lines = norm_lines[st:ed]
        block = "\n".join([x for x in block_lines if x]).strip()
        if not block:
            continue

        title = block_lines[0] if block_lines else ""
        if len(block_lines) >= 2 and block_lines[1] == block_lines[0]:
            title = block_lines[0]

        title = normalize_text_basic(title)
        if is_bad_title(title, min_title_len, max_title_len):
            inferred = infer_title_from_body(block)
            if inferred and not is_bad_title(inferred, min_title_len, max_title_len):
                title = inferred
            else:
                title = "UNKNOWN_TITLE"

        items.append({
            "标题": title,
            "起始页": -1,
            "结束页": -1,
            "正文": block,
        })

    return items

def extract_factiva_rtf_to_df(
    rtf_path: str,
    min_title_len: int = DEFAULT_MIN_TITLE_LEN,
    max_title_len: int = DEFAULT_MAX_TITLE_LEN,
) -> pd.DataFrame:
    with open(rtf_path, "rb") as f:
        raw = f.read()

    text = ""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if not text:
        text = raw.decode("latin-1", errors="ignore")

    plain = rtf_to_text(text)
    blocks = segment_factiva_articles_from_text(plain, min_title_len=min_title_len, max_title_len=max_title_len)

    rows = []
    for b in blocks:
        rows.append({
            "标题": b["标题"],
            "起始页": b["起始页"],
            "结束页": b["结束页"],
            "正文": b["正文"],
            "source_file": os.path.basename(rtf_path),
            "source_path": os.path.abspath(rtf_path),
            "source_type": "rtf",
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3) Scheme C scorer (Rule + LLM + cache)
# =============================================================================

@dataclass
class Article:
    source_csv: str
    source_file: str
    source_type: str

    title: str
    start_page: int
    end_page: int
    text: str

    publisher: str
    publisher_norm: str
    pub_date: str  # YYYY-MM-DD or UNKNOWN
    pub_time: str  # HH:MM or ""
    pub_time_alt: str  # optional, when both header/footer exist and differ
    content_sha1: str


# ----------------------------
# Dictionaries (as provided; kept stable)
# ----------------------------

CRASH_1_CN = ["回调","调整","震荡","震荡加剧","波动加剧","波动率上升","短期承压","面临压力","走低","小幅走低","小幅回落","略有下跌","获利回吐","技术性回调","风险偏好降温","风险偏好降至谨慎"]
CRASH_1_EN = ["correction","pullback","technical pullback","volatility rises","volatility picks up","higher volatility","faces pressure","under pressure","comes under pressure","edges lower","drifts lower","moves slightly lower","mild decline","modest decline","small loss","slight loss","profit taking","profit-taking pressure","risk appetite cools","risk appetite turns cautious"]

CRASH_2_CN = ["暴跌","大跌","重挫","急跌","急挫","大幅下跌","大幅跳水","狂跌","跌势加剧","持续下挫","抛售潮","大规模抛售","大规模抛盘","恐慌性抛售","跌入熊市","步入熊市","跌入技术性熊市","熊市担忧","对熊市的担忧加剧"]
CRASH_2_EN = ["plunge","plunges","plunged","tumble","tumbles","tumbled","slump","slumps","slumped","sharp drop","sharp fall","sharp losses","steep decline","steep losses","big drop","big fall","heavy losses","heavy selling","broad-based selloff","broad-based losses","broad-based selling","waves of selling","selling wave","stocks enter bear market","enters a bear market","falls into a bear market","slips into bear territory","bear-market worries","growing fears of a bear market"]

CRASH_3_CN = ["崩盘","股灾","市场崩溃","金融体系崩溃","灾难性下跌","灾难性暴跌","灾难性抛售","史上最惨一日","史上最惨一周","创历史最大跌幅","创历史性暴跌","比 2008 年更严重","比金融危机时期更严重","系统性危机爆发","系统性风险全面爆发"]
CRASH_3_EN = ["market crash","stock market crash","meltdown","market meltdown","financial meltdown","market collapse","collapse in markets","collapse in stock prices","systemic collapse","collapse of the financial system","catastrophic drop","catastrophic losses","catastrophic selloff","worst crash since 2008","worst crash since the financial crisis","biggest crash in history","record-breaking crash","panic selling across markets","global market bloodbath","systemic crisis erupts","systemic crisis unfolds"]

CERT0_CN = ["可能","或将","或许","有风险","存在","不排除","如果"]
CERT0_EN = ["may","might","could","risk of","poses a risk","there is a chance","it is possible","some worry","some fear","if "]

CERT1_CN = ["很可能","大概率","预计会","料将","市场普遍担心","被认为可能"]
CERT1_EN = ["is likely to","is expected to","is set to","is poised to","could well","is seen as likely","growing concern","increasingly expect"]

CERT2_CN = ["正在崩盘","已经崩盘","正在经历股灾","几乎可以肯定","几乎不可避免","必然","除非"]
CERT2_EN = ["is crashing","has crashed","in the middle of a crash","almost certain","inevitable","virtually certain","is now in meltdown","underway"]

FOMO_1_CN = ["情绪转暖","情绪好转","市场情绪改善","风险偏好回升","风险偏好上升","看多情绪升温","做多意愿提升"]
FOMO_1_EN = ["sentiment improves","market sentiment turns positive","risk appetite returns","risk-on mood returns","bullish sentiment rises","turn more bullish"]

FOMO_2_CN = ["害怕踏空","担心踏空","恐惧踏空","害怕错过行情","担心错过机会","追涨","追高","抢筹","蜂拥买入","扎堆买入","一窝蜂买入","排队入场","散户排队进场"]
FOMO_2_EN = ["fear of missing out","FOMO","fear of missing the rally","fear of being left behind","chasing the rally","chasing gains","chasing momentum","pile into","piling into","rush to buy","flocking into","crowding into","scramble to buy","scrambling to get exposure"]

FOMO_3_CN = ["投机狂潮","投机热潮","泡沫化上涨","明显泡沫","泡沫风险攀升","疯狂买入","疯狂追涨","恐慌式买入","散户疯狂涌入","融资杠杆激增","成交量爆炸式增长","成交量创历史新高"]
FOMO_3_EN = ["speculative mania","speculative frenzy","bubble-like rally","clear signs of a bubble","buying frenzy","buying spree","panic buying","retail stampede","leveraged buying","margin debt surges","trading volume explodes","record inflows"]

FOMO_CERT0_CN = ["似乎受到 FOMO 影响","可能存在一定 FOMO 情绪","部分投资者看起来是在追涨","有观点认为"]
FOMO_CERT0_EN = ["may be driven in part by FOMO","seems to be influenced by FOMO","there may be some fear of missing out","some observers suggest"]

FOMO_CERT1_CN = ["很大程度上由 FOMO 驱动","FOMO 是当前上涨的重要因素","追涨和害怕踏空","源于对踏空的担忧"]
FOMO_CERT1_EN = ["largely driven by FOMO","a major driver","significantly influenced by fear of missing out","fear of being left behind is an important factor"]

FOMO_CERT2_CN = ["几乎完全由 FOMO 驱动","本质上是踏空恐惧推动","主要源于害怕踏空","没有 FOMO 很难维持"]
FOMO_CERT2_EN = ["almost entirely FOMO-driven","driven overwhelmingly by fear of missing out","mainly by FOMO rather than fundamentals","without FOMO, this rally would likely not exist"]

RE_PCT = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s?%")
RE_DAY_DROP_CN = re.compile(r"(单日|一天|日内).{0,12}(下跌|跌|暴跌|大跌).{0,12}(\d{1,2}(?:\.\d+)?)\s?%")
RE_DAY_DROP_EN = re.compile(r"(falls|fell|plunges|plunged|tumbles|tumbled|slides|slid)\s+(\d{1,2}(?:\.\d+)?)\s?%")
RE_WIPEOUT_EN = re.compile(r"(wipes out|erases|loses)\s+\$?\s?(\d+(?:\.\d+)?)\s*(billion|million)", re.IGNORECASE)
RE_WIPEOUT_CN = re.compile(r"(市值蒸发|市值缩水|蒸发).{0,12}(\d+).{0,8}(亿|百亿|千亿|万亿)")

def contains_any(text_lc: str, phrases: List[str]) -> bool:
    for p in phrases:
        if p and p.lower() in text_lc:
            return True
    return False


# =============================================================================
# 3.1) Publisher/date/time parsing (robust)
# =============================================================================

RE_DT_CN = re.compile(
    r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日(?:\s*(?P<h>\d{1,2})\s*:\s*(?P<mi>\d{2}))?"
)
RE_DT_EN = re.compile(
    r"(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+(?P<d>\d{1,2}),\s*(?P<y>\d{4})\s+(?P<h>\d{1,2}):(?P<mi>\d{2})",
    re.IGNORECASE
)

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

KNOWN_PUBLISHERS = {
    "Dow Jones Institutional News",
    "Dow Jones Newswires",
    "The Wall Street Journal",
    "Barron's",
    "MarketWatch",
}

def normalize_publisher(p: str) -> str:
    p = normalize_text_basic(p)
    p = re.sub(r"^\(END\)\s*", "", p, flags=re.IGNORECASE).strip()
    p = re.sub(r"^\(FROM\s+.*?\)\s*", "", p, flags=re.IGNORECASE).strip()
    p = re.sub(r"\s+", " ", p).strip()
    return p or "UNKNOWN"

def _is_noise_line_for_publisher(ln: str) -> bool:
    s = normalize_text_basic(ln)
    if not s:
        return True
    if re.match(r"(?i)^\(END\)", s) or re.match(r"(?i)^\(FROM\)", s):
        return True
    if s in ("英文", "中文", "English", "Chinese"):
        return True
    if re.fullmatch(r"[A-Z0-9]{2,10}", s):
        return True
    if s.startswith("文件") or re.match(r"(?i)^file\b", s):
        return True
    if re.search(r"(?i)^\d[\d,]*\s*(字|words?)\s*$", s):
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    return False

def pick_publisher(lines: List[str]) -> str:
    cleaned = [normalize_text_basic(ln) for ln in lines if normalize_text_basic(ln)]

    for ln in cleaned[:250]:
        if ln in KNOWN_PUBLISHERS:
            return ln

    candidates: List[str] = []
    for ln in (cleaned[:350] + cleaned[-350:]):
        if _is_noise_line_for_publisher(ln):
            continue
        if "Dow Jones" in ln:
            candidates.append(ln)

    if candidates:
        for ln in candidates:
            if 8 <= len(ln) <= 80:
                return ln
        return candidates[0]

    for ln in cleaned[:250]:
        if _is_noise_line_for_publisher(ln):
            continue
        if 8 <= len(ln) <= 80 and not re.search(r"@", ln):
            return ln

    return "UNKNOWN"

def parse_pub_meta_from_text(text: str) -> Tuple[str, str, str, str]:
    """
    Returns (publisher, pub_date, pub_time, pub_time_alt)
    """
    t = normalize_text_basic(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    publisher = normalize_publisher(pick_publisher(lines))

    cn_date, cn_time = "UNKNOWN", ""
    en_date, en_time = "UNKNOWN", ""

    for ln in lines[:250]:
        m = RE_DT_CN.search(ln)
        if m:
            y = int(m.group("y"))
            mo = int(m.group("m"))
            d = int(m.group("d"))
            cn_date = f"{y:04d}-{mo:02d}-{d:02d}"
            hh = m.group("h"); mi = m.group("mi")
            if hh and mi:
                cn_time = f"{int(hh):02d}:{int(mi):02d}"
            break

    for ln in lines[-300:]:
        m = RE_DT_EN.search(ln)
        if m:
            y = int(m.group("y"))
            mon = m.group("mon").strip().lower()
            key = mon if mon in MONTH_MAP else mon[:3]
            mo = MONTH_MAP.get(key, 0)
            d = int(m.group("d"))
            hh = int(m.group("h"))
            mi = int(m.group("mi"))
            if mo:
                en_date = f"{y:04d}-{mo:02d}-{d:02d}"
                en_time = f"{hh:02d}:{mi:02d}"
                break

    pub_date = cn_date if cn_date != "UNKNOWN" else en_date
    pub_time = cn_time if cn_time else en_time

    pub_time_alt = ""
    if pub_date != "UNKNOWN":
        if cn_date == pub_date and en_date == pub_date and cn_time and en_time and cn_time != en_time:
            pub_time = cn_time
            pub_time_alt = en_time

    if pub_date == "UNKNOWN" and en_date != "UNKNOWN":
        pub_date = en_date
    if not pub_time and en_time:
        pub_time = en_time

    return publisher, pub_date, pub_time, pub_time_alt


# =============================================================================
# 3.2) Rule scoring
# =============================================================================

def direction_rule(text_lc: str) -> Tuple[int, int]:
    crash_dir = 0
    if (
        contains_any(text_lc, CRASH_1_CN + CRASH_2_CN + CRASH_3_CN) or
        contains_any(text_lc, CRASH_1_EN + CRASH_2_EN + CRASH_3_EN) or
        re.search(r"(selloff|sell-off|risk-off|panic selling|bear market|bear territory|market rout)", text_lc) or
        RE_DAY_DROP_CN.search(text_lc) or
        RE_DAY_DROP_EN.search(text_lc) or
        RE_WIPEOUT_EN.search(text_lc) or
        RE_WIPEOUT_CN.search(text_lc)
    ):
        crash_dir = 1

    fomo_dir = 1 if (
        contains_any(text_lc, FOMO_1_CN + FOMO_2_CN + FOMO_3_CN) or
        contains_any(text_lc, FOMO_1_EN + FOMO_2_EN + FOMO_3_EN)
    ) else 0

    return crash_dir, fomo_dir

def crash_intensity_rule(text_lc: str) -> int:
    if contains_any(text_lc, CRASH_3_CN) or contains_any(text_lc, CRASH_3_EN):
        return 3
    if "biggest one-day" in text_lc or "worst crash" in text_lc or "black monday" in text_lc:
        return 3

    if contains_any(text_lc, CRASH_2_CN) or contains_any(text_lc, CRASH_2_EN):
        return 2
    if RE_DAY_DROP_CN.search(text_lc) or RE_DAY_DROP_EN.search(text_lc) or RE_WIPEOUT_EN.search(text_lc) or RE_WIPEOUT_CN.search(text_lc):
        return 2
	
    m = RE_PCT.findall(text_lc)
    if m:
        try:
            max_pct = max(float(x) for x in m)
            if max_pct >= 2.0 and re.search(
                r"(跌|下跌|暴跌|大跌|plung|tumble|slump|selloff|sell-off|drop|fall|fell|slides?|slid|lower|declin(ed|es)?)",
                text_lc
            ):
                return 2
        except Exception:
            pass

    if contains_any(text_lc, CRASH_1_CN) or contains_any(text_lc, CRASH_1_EN):
        return 1

    return 0


def crash_certainty_rule(text_lc: str, intensity: int) -> int:
    if contains_any(text_lc, CERT2_CN) or contains_any(text_lc, CERT2_EN):
        return 2
    if intensity >= 2 and re.search(r"\b(has|have)\s+(plunged|tumbled|slumped|crashed)\b", text_lc):
        return 2
    if intensity >= 2 and ("已经" in text_lc or "已" in text_lc) and ("暴跌" in text_lc or "大跌" in text_lc or "重挫" in text_lc):
        return 2

    if contains_any(text_lc, CERT1_CN) or contains_any(text_lc, CERT1_EN):
        return 1
    if contains_any(text_lc, CERT0_CN) or contains_any(text_lc, CERT0_EN):
        return 0

    if intensity >= 2:
        return 1
    return 0

def fomo_intensity_rule(text_lc: str) -> int:
    level = 0

    if contains_any(text_lc, FOMO_3_CN) or contains_any(text_lc, FOMO_3_EN):
        level = max(level, 3)

    if (
        contains_any(text_lc, FOMO_2_CN) or
        contains_any(text_lc, FOMO_2_EN) or
        re.search(r"\bbuy(ing)? the dip\b|\bdip[- ]buying\b", text_lc) or
        re.search(r"\bsnap(s|ped|ping)? up\b", text_lc) or
        re.search(r"\bchasing (the )?rally\b", text_lc)
    ):
        level = max(level, 2)

    if (
        contains_any(text_lc, FOMO_1_CN) or
        contains_any(text_lc, FOMO_1_EN) or
        re.search(r"\brisk[- ]on\b", text_lc) or
        re.search(r"risk appetite (returns|improves|picks up)", text_lc)
    ):
        level = max(level, 1)

    return level


def fomo_certainty_rule(text_lc: str, intensity: int) -> int:
    if contains_any(text_lc, FOMO_CERT2_CN) or contains_any(text_lc, FOMO_CERT2_EN):
        return 2
    if contains_any(text_lc, FOMO_CERT1_CN) or contains_any(text_lc, FOMO_CERT1_EN):
        return 1
    if contains_any(text_lc, FOMO_CERT0_CN) or contains_any(text_lc, FOMO_CERT0_EN):
        return 0
    if intensity >= 2:
        return 1
    return 0

def score_rule(a: Article) -> Dict:
    text_lc = (a.title + "\n" + a.text).lower()

    ic = crash_intensity_rule(text_lc)
    iF = fomo_intensity_rule(text_lc)

    dir_c_raw, dir_f_raw = direction_rule(text_lc)

    dir_c = 1 if (ic > 0 or dir_c_raw == 1) else 0
    dir_f = 1 if (iF > 0 or dir_f_raw == 1) else 0

    if dir_c == 1:
        cc = crash_certainty_rule(text_lc, ic)
    else:
        ic, cc = 0, 0 

    if dir_f == 1:
        cF = fomo_certainty_rule(text_lc, iF)
    else:
        iF, cF = 0, 0

    out = {
        "dir_crash": dir_c,
        "intensity_crash": ic,
        "certainty_crash": cc,
        "dir_fomo": dir_f,
        "intensity_fomo": iF,
        "certainty_fomo": cF,
        "crash_score": compute_score(dir_c, ic, cc),
        "fomo_score": compute_score(dir_f, iF, cF),
    }
    return out


# =============================================================================
# 3.3) LLM scoring + cache  (SPEED-OPTIMIZED)
# =============================================================================

LLM_SYSTEM_RUBRIC = """
You are scoring a financial news article on two independent dimensions: Crash and Fomo.

Return a SINGLE JSON object strictly following this schema:
{
  "dir_crash": 0 or 1,
  "intensity_crash": 0..3,
  "certainty_crash": 0..2,
  "dir_fomo": 0 or 1,
  "intensity_fomo": 0..3,
  "certainty_fomo": 0..2,
  "evidence": { "crash": [<=2 short quotes], "fomo": [<=2 short quotes] }
}

General rules:
- Default is NO signal: if you do not find clear textual evidence for Crash or Fomo, keep dir_* = 0, intensity_* = 0, certainty_* = 0.
- If dir_crash=0 then intensity_crash=0 and certainty_crash=0.
- If dir_fomo=0 then intensity_fomo=0 and certainty_fomo=0.
- Evidence quotes must be short (<= 20 words each) and copied verbatim from the article, focusing on the key phrases that justify the label.

CRASH dimension (price DOWN / fear of losses):
- dir_crash=1 only if the article clearly talks about meaningful downside risk or realized downside moves for indexes or broad markets (not just one small stock).
- intensity_crash:
  0 = no meaningful downside signal, normal volatility, or mixed/neutral outlook.
  1 = mild decline / pressure / pullback / correction language, small or short-term losses, "under pressure", "edges lower".
  2 = sharp drop / heavy selling / broad-based selloff / bear market worries, large one-day or multi-day losses, big percentage drops, "selloff", "tumbles", "bear market fears".
  3 = crash / meltdown / systemic crisis language, comparisons to 2008 or historic crashes, "market crash", "meltdown", "systemic crisis", "bloodbath".
- certainty_crash:
  0 = very speculative / conditional / possible only ("may", "might", "risk of", "could", "if ...").
  1 = likely/expected in the near future but not clearly realized yet ("is likely to", "is expected to", "is set to").
  2 = already happening or almost certain / unavoidable (describing realized crashes or very strong language like "has crashed", "is crashing", "inevitable").

FOMO dimension (risk-on buying / chasing gains / fear of missing out):
- Fomo in this task is intentionally **broad and lenient**:
  - It covers explicit “fear of missing out” / “chasing the rally” type language, **and**
  - More general risk-on / buy-the-dip / crowded trades where investors are clearly eager to gain upside.
- dir_fomo = 1 if the article clearly describes any of the following for equities or other risky assets (broad market, sectors, or popular stocks):
  - Investors are buying aggressively, rotating into risk assets, "piling into" hot themes, or "rushing to buy".
  - A clearly bullish, risk-on tone with strong or repeated buy recommendations, substantial inflows, or overweight positions in risky assets.
  - Persistent “search for yield” or “reach for risk” behavior where investors move from safe assets into riskier ones because they do not want to miss potential gains.
- intensity_fomo:
  0 = no Fomo or risk-on signal: neutral/mixed tone, or only factual price moves with no sense of investors actively seeking upside.
  1 = moderate risk-on / clear buying appetite, but not a frenzy.
      Examples:
      - "investors continue to buy tech stocks",
      - "funds flow into equities as risk appetite improves",
      - "analysts broadly recommend buying on recent weakness",
      - "investors rotate out of bonds into stocks".
      Use 1 whenever there is a clear tilt toward buying / risk-on behavior or repeated buy recommendations, even if there is no explicit 'FOMO' wording.
  2 = strong Fomo / chasing behavior.
      Typical patterns:
      - explicit "fear of missing out", "fear of being left behind",
      - "pile into", "rush to buy", "scramble to get exposure",
      - "chasing the rally", "crowding into AI stocks" etc.
      The tone is that investors are actively chasing recent gains or hot themes.
  3 = frenzy / bubble / speculative mania.
      Language like:
      - "buying frenzy", "panic buying", "speculative mania", "bubble-like rally",
      - "retail stampede", record leverage or record trading volumes clearly framed as chasing rather than fundamentals.
- certainty_fomo:
  0 = weak or tentative reference to Fomo / risk-on behavior:
      e.g., "may be driven by FOMO", "some observers suggest investors are chasing".
  1 = Fomo or risk-on behavior is described as an important or major driver among others:
      e.g., "largely driven by FOMO", "fear of being left behind is a major factor", "strong inflows show growing risk appetite".
  2 = Fomo or chasing behavior is described as the dominant driver:
      e.g., "almost entirely FOMO-driven", "overwhelmingly driven by fear of missing out", "without FOMO this rally would likely not exist".

Interaction between Crash and Fomo:
- It is possible (and allowed) that both Crash and Fomo are non-zero (for example: a fragile boom where investors chase a rally while fearing a crash, or buy-the-dip behavior after a selloff).
- However, if the article mainly describes a severe crash, meltdown, or systemic crisis, and does NOT clearly describe investors buying or chasing anything, set dir_fomo=0.
- Similarly, if the article mainly describes a strong Fomo-driven rally without meaningful downside language, set dir_crash=0.

Output JSON only. No extra text.
""".strip()


def build_llm_user_prompt(title: str, text: str, max_chars: int) -> str:
    """
    Dynamic part only (better for prefix caching):
    - Keep rubric in system prompt.
    - Truncate article text aggressively for latency control.
    """
    text = normalize_text_basic(text)
    max_chars = int(max_chars) if max_chars is not None else 4000
    if max_chars > 0 and len(text) > max_chars:
        # head + tail (keeps context + footer meta where evidence may live)
        head = int(max_chars * 0.70)
        tail = max_chars - head
        text = text[:head] + "\n...\n" + text[-tail:]

    return f"""TITLE:
{title}

ARTICLE:
{text}
"""

def llm_call_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    thinking: str,
    base_url: Optional[str],
    force_json_object: bool,
) -> str:

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    env_base = (os.environ.get("OPENAI_BASE_URL", "").strip() or None)
    final_base = (base_url.strip() if base_url else None) or env_base or "https://api.deepseek.com"

    client = OpenAI(api_key=api_key, base_url=final_base)

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=int(max_tokens),
    )

    if force_json_object:
        kwargs["response_format"] = {"type": "json_object"}

    if thinking in ("disabled", "enabled") and "deepseek" in (final_base or "").lower():
        kwargs["extra_body"] = {"thinking": {"type": thinking}}

    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except TypeError:
        pass
    except Exception as e:
        last_exc = e
        try:
            kwargs.pop("response_format", None)
            kwargs.pop("extra_body", None)
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception:
            raise last_exc

    kwargs.pop("response_format", None)
    kwargs.pop("extra_body", None)
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""



def parse_llm_json(raw: str) -> Dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError("LLM output is not valid JSON and no JSON block found.")
        return json.loads(m.group(0))

def coerce_labels(d: Dict) -> Dict:
    out = {
        "dir_crash": clamp_int(d.get("dir_crash", 0), 0, 1),
        "intensity_crash": clamp_int(d.get("intensity_crash", 0), 0, 3),
        "certainty_crash": clamp_int(d.get("certainty_crash", 0), 0, 2),
        "dir_fomo": clamp_int(d.get("dir_fomo", 0), 0, 1),
        "intensity_fomo": clamp_int(d.get("intensity_fomo", 0), 0, 3),
        "certainty_fomo": clamp_int(d.get("certainty_fomo", 0), 0, 2),
        "evidence": d.get("evidence", {}) or {},
    }
    if out["dir_crash"] == 0:
        out["intensity_crash"] = 0
        out["certainty_crash"] = 0
    if out["dir_fomo"] == 0:
        out["intensity_fomo"] = 0
        out["certainty_fomo"] = 0

    out["crash_score"] = compute_score(out["dir_crash"], out["intensity_crash"], out["certainty_crash"])
    out["fomo_score"] = compute_score(out["dir_fomo"], out["intensity_fomo"], out["certainty_fomo"])
    return out

class LLMCache:
    """SQLite cache keyed by (model_id, content_sha1). Thread-safe for optional concurrency."""
    def __init__(self, path: str):
        self.path = path
        ensure_dir_for_file(path)
        self._lock = threading.Lock()
        # check_same_thread=False to allow read/write from ThreadPoolExecutor
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
              model_id TEXT NOT NULL,
              content_sha1 TEXT NOT NULL,
              raw TEXT NOT NULL,
              json TEXT NOT NULL,
              PRIMARY KEY (model_id, content_sha1)
            )
            """)
            self.conn.commit()

    def get(self, model_id: str, content_sha1: str) -> Optional[Dict]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT raw, json FROM llm_cache WHERE model_id=? AND content_sha1=?",
                        (model_id, content_sha1))
            row = cur.fetchone()
        if not row:
            return None
        raw, js = row
        d = json.loads(js)
        d["_raw"] = raw
        d["_cache_hit"] = True
        return d

    def set(self, model_id: str, content_sha1: str, raw: str, d: Dict):
        payload = json.dumps(d, ensure_ascii=False)
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO llm_cache(model_id, content_sha1, raw, json) VALUES(?,?,?,?)",
                (model_id, content_sha1, raw, payload)
            )
            self.conn.commit()

def score_llm(
    a: Article,
    model: str,
    cache: Optional[LLMCache],
    force_llm: bool,
    max_chars: int,
    max_tokens: int,
    thinking: str,
    base_url: Optional[str],
    force_json_object: bool,
) -> Dict:
    model_id = f"openai_compat:{model}"
    if cache and not force_llm:
        hit = cache.get(model_id, a.content_sha1)
        if hit:
            return hit

    user_prompt = build_llm_user_prompt(a.title, a.text, max_chars=max_chars)
    raw = llm_call_openai_compatible(
        system_prompt=LLM_SYSTEM_RUBRIC,
        user_prompt=user_prompt,
        model=model,
        max_tokens=max_tokens,
        thinking=thinking,
        base_url=base_url,
        force_json_object=force_json_object,
    )
    parsed = parse_llm_json(raw)
    coerced = coerce_labels(parsed)
    coerced["_raw"] = (raw or "").strip()
    coerced["_cache_hit"] = False

    if cache:
        cache.set(model_id, a.content_sha1, coerced["_raw"], coerced)

    return coerced


# =============================================================================
# 3.4) Scheme C fusion + dedup + aggregation
# =============================================================================

def fuse_scores(rule: Dict, llm: Dict, weight_rule: float, agree_tol: float) -> Dict:
    wR = weight_rule
    wL = 1.0 - weight_rule

    r_c, l_c = float(rule["crash_score"]), float(llm["crash_score"])
    if abs(r_c - l_c) <= agree_tol:
        fused_c, mode_c, disagree_c = 0.5 * (r_c + l_c), "avg", 0
    else:
        fused_c, mode_c, disagree_c = wR * r_c + wL * l_c, "weighted", 1

    r_f, l_f = float(rule["fomo_score"]), float(llm["fomo_score"])
    if abs(r_f - l_f) <= agree_tol:
        fused_f, mode_f, disagree_f = 0.5 * (r_f + l_f), "avg", 0
    else:
        fused_f, mode_f, disagree_f = wR * r_f + wL * l_f, "weighted", 1

    dir_conflict_crash = int(rule["dir_crash"] != llm["dir_crash"])
    dir_conflict_fomo = int(rule["dir_fomo"] != llm["dir_fomo"])
    needs_review = int(dir_conflict_crash or dir_conflict_fomo or disagree_c or disagree_f)

    return {
        "fused_crash_score": max(0.0, min(1.0, fused_c)),
        "fused_fomo_score": max(0.0, min(1.0, fused_f)),
        "fuse_mode_crash": mode_c,
        "fuse_mode_fomo": mode_f,
        "dir_conflict_crash": dir_conflict_crash,
        "dir_conflict_fomo": dir_conflict_fomo,
        "disagree_crash": disagree_c,
        "disagree_fomo": disagree_f,
        "absdiff_crash": abs(r_c - l_c),
        "absdiff_fomo": abs(r_f - l_f),
        "needs_review": needs_review,
    }

def dedup_articles(articles: List[Article]) -> List[Article]:
    """Dedup key requirement: (publisher, date, content_sha1). Here publisher_norm is used for stability."""
    seen = set()
    out = []
    for a in articles:
        k = (a.publisher_norm, a.pub_date, a.content_sha1)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out

def aggregate_daily(rows: List[Dict], crash_key: str, fomo_key: str) -> pd.DataFrame:
    df = pd.DataFrame([{"Date": r["date"], "Crash_t": r[crash_key], "Fomo_t": r[fomo_key]} for r in rows])
    if df.empty:
        return pd.DataFrame(columns=["Date", "Crash_t", "Fomo_t"])
    agg = df.groupby("Date", as_index=False)[["Crash_t", "Fomo_t"]].sum()

    def key_dt(x: str):
        try:
            return datetime.strptime(x, "%Y-%m-%d")
        except Exception:
            return datetime.max

    return agg.sort_values("Date", key=lambda s: s.map(key_dt)).reset_index(drop=True)


# =============================================================================
# 4) Load extracted DF/CSV -> Article list
# =============================================================================

def load_articles_from_df(df: pd.DataFrame, source_csv: str) -> List[Article]:
    expected = ["标题", "起始页", "结束页", "正文"]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Got columns: {list(df.columns)}")

    articles: List[Article] = []
    for _, r in df.iterrows():
        title = normalize_text_basic(str(r["标题"]))
        text = normalize_text_basic(str(r["正文"]))

        try:
            sp = int(r["起始页"])
        except Exception:
            sp = -1
        try:
            ep = int(r["结束页"])
        except Exception:
            ep = -1

        publisher, pub_date, pub_time, pub_time_alt = parse_pub_meta_from_text(text)
        publisher_norm = normalize_publisher(publisher)

        sha = sha1_text(title, text)

        articles.append(Article(
            source_csv=source_csv,
            source_file=str(r.get("source_file", "")) if "source_file" in df.columns else "",
            source_type=str(r.get("source_type", "")) if "source_type" in df.columns else "",
            title=title,
            start_page=sp,
            end_page=ep,
            text=text,
            publisher=publisher,
            publisher_norm=publisher_norm,
            pub_date=pub_date,
            pub_time=pub_time,
            pub_time_alt=pub_time_alt,
            content_sha1=sha,
        ))
    return articles

def load_articles_from_csv(input_csv: str) -> List[Article]:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    return load_articles_from_df(df, source_csv=input_csv)


# =============================================================================
# 5) Extraction orchestrator (batch)
# =============================================================================

def extract_inputs_to_df(
    input_pdf: str,
    input_rtf: str,
    input_csv: str,
    input_dir: str,
    toc_scan_pages: int,
    toc_dot_threshold: int,
    min_title_len: int,
    max_title_len: int,
    max_page_gap: Optional[int],
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    if input_csv:
        df = pd.read_csv(input_csv, encoding="utf-8-sig")
        if "source_file" not in df.columns:
            df["source_file"] = os.path.basename(input_csv)
        if "source_path" not in df.columns:
            df["source_path"] = os.path.abspath(input_csv)
        if "source_type" not in df.columns:
            df["source_type"] = "csv"
        dfs.append(df)

    if input_pdf:
        df = extract_factiva_pdf_to_df(
            pdf_path=input_pdf,
            toc_scan_pages=toc_scan_pages,
            toc_dot_threshold=toc_dot_threshold,
            min_title_len=min_title_len,
            max_title_len=max_title_len,
            max_page_gap=max_page_gap,
        )
        dfs.append(df)

    if input_rtf:
        df = extract_factiva_rtf_to_df(
            rtf_path=input_rtf,
            min_title_len=min_title_len,
            max_title_len=max_title_len,
        )
        dfs.append(df)

    if input_dir:
        pdfs = list_files(input_dir, (".pdf",))
        rtfs = list_files(input_dir, (".rtf",))
        csvs = list_files(input_dir, (".csv",))

        for p in pdfs:
            try:
                df = extract_factiva_pdf_to_df(
                    pdf_path=p,
                    toc_scan_pages=toc_scan_pages,
                    toc_dot_threshold=toc_dot_threshold,
                    min_title_len=min_title_len,
                    max_title_len=max_title_len,
                    max_page_gap=max_page_gap,
                )
                dfs.append(df)
                print(f"[OK] Extracted PDF: {p} | articles={len(df)}")
            except Exception as e:
                print(f"[WARN] Failed PDF extraction: {p} | {e}")

        for p in rtfs:
            try:
                df = extract_factiva_rtf_to_df(
                    rtf_path=p,
                    min_title_len=min_title_len,
                    max_title_len=max_title_len,
                )
                dfs.append(df)
                print(f"[OK] Extracted RTF: {p} | articles={len(df)}")
            except Exception as e:
                print(f"[WARN] Failed RTF extraction: {p} | {e}")

        for p in csvs:
            try:
                df = pd.read_csv(p, encoding="utf-8-sig")
                if all(c in df.columns for c in ["标题", "起始页", "结束页", "正文"]):
                    if "source_file" not in df.columns:
                        df["source_file"] = os.path.basename(p)
                    if "source_path" not in df.columns:
                        df["source_path"] = os.path.abspath(p)
                    if "source_type" not in df.columns:
                        df["source_type"] = "csv"
                    dfs.append(df)
                    print(f"[OK] Loaded CSV: {p} | rows={len(df)}")
            except Exception:
                continue

    if not dfs:
        raise ValueError("No valid inputs found. Provide --input_dir or one of --input_pdf/--input_rtf/--input_csv.")

    all_df = pd.concat(dfs, ignore_index=True)

    for c in ["标题", "起始页", "结束页", "正文"]:
        if c not in all_df.columns:
            raise ValueError(f"Combined extraction missing required column: {c}")

    for c in ["source_file", "source_path", "source_type"]:
        if c not in all_df.columns:
            all_df[c] = ""

    return all_df


# =============================================================================
# 6) Run scoring pipeline
# =============================================================================

def run_pipeline(
    extracted_df: pd.DataFrame,
    extracted_csv_out: str,
    output_xlsx: str,
    model: str,
    cache_db: str,
    weight_rule: float,
    agree_tol: float,
    keep_llm_raw: bool,
    disable_llm: bool,
    force_llm: bool,
    # speed knobs
    max_chars: int,
    max_tokens: int,
    thinking: str,
    base_url: str,
    force_json_object: bool,
    workers: int,
) -> None:
    if extracted_csv_out:
        ensure_dir_for_file(extracted_csv_out)
        extracted_df.to_csv(extracted_csv_out, index=False, encoding="utf-8-sig")
        print(f"[OK] Combined extracted CSV written: {extracted_csv_out} | rows={len(extracted_df)}")

    articles = load_articles_from_df(extracted_df, source_csv=(extracted_csv_out or "combined_df"))
    if not articles:
        raise RuntimeError("No articles loaded after extraction.")

    deduped = dedup_articles(articles)
    cache = None if disable_llm else (LLMCache(cache_db) if cache_db else None)

    rows_by_idx: Dict[int, Dict] = {}
    n_fail = 0
    n_cache_hit = 0
    n_api_call = 0

    def process_one(idx: int, a: Article) -> Tuple[int, Dict, Dict]:
        """
        Returns: (idx, row, stats)
          stats: {"cache_hit":0/1,"api_call":0/1,"failed":0/1}
        """
        rule = score_rule(a)

        if disable_llm:
            llm = rule.copy()
            llm["evidence"] = {"crash": [], "fomo": []}
            llm["_raw"] = "[LLM DISABLED]"
            llm["_cache_hit"] = False
            stats = {"cache_hit": 0, "api_call": 0, "failed": 0}
        else:
            try:
                llm = score_llm(
                    a,
                    model=model,
                    cache=cache,
                    force_llm=force_llm,
                    max_chars=max_chars,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    base_url=base_url,
                    force_json_object=force_json_object,
                )
                stats = {
                    "cache_hit": 1 if llm.get("_cache_hit") else 0,
                    "api_call": 0 if llm.get("_cache_hit") else 1,
                    "failed": 0,
                }
            except Exception as e:
                llm = rule.copy()
                llm["evidence"] = {"crash": [], "fomo": []}
                llm["_raw"] = f"[LLM FAILED] {e}"
                llm["_cache_hit"] = False
                stats = {"cache_hit": 0, "api_call": 0, "failed": 1}

        fused = fuse_scores(rule, llm, weight_rule=weight_rule, agree_tol=agree_tol)

        row = {
            "source_csv": a.source_csv,
            "source_file": a.source_file,
            "source_type": a.source_type,

            "title": a.title,
            "start_page": a.start_page,
            "end_page": a.end_page,

            "publisher": a.publisher,
            "publisher_norm": a.publisher_norm,
            "date": a.pub_date,
            "time": a.pub_time,
            "time_alt": a.pub_time_alt,
            "content_sha1": a.content_sha1,
            "text_len": len(a.text),

            # rule
            "rule_dir_crash": rule["dir_crash"],
            "rule_intensity_crash": rule["intensity_crash"],
            "rule_certainty_crash": rule["certainty_crash"],
            "rule_crash_score": rule["crash_score"],
            "rule_dir_fomo": rule["dir_fomo"],
            "rule_intensity_fomo": rule["intensity_fomo"],
            "rule_certainty_fomo": rule["certainty_fomo"],
            "rule_fomo_score": rule["fomo_score"],

            # llm
            "llm_dir_crash": llm.get("dir_crash", 0),
            "llm_intensity_crash": llm.get("intensity_crash", 0),
            "llm_certainty_crash": llm.get("certainty_crash", 0),
            "llm_crash_score": llm.get("crash_score", 0.0),
            "llm_dir_fomo": llm.get("dir_fomo", 0),
            "llm_intensity_fomo": llm.get("intensity_fomo", 0),
            "llm_certainty_fomo": llm.get("certainty_fomo", 0),
            "llm_fomo_score": llm.get("fomo_score", 0.0),

            # fused
            "fused_crash_score": fused["fused_crash_score"],
            "fused_fomo_score": fused["fused_fomo_score"],
            "fuse_mode_crash": fused["fuse_mode_crash"],
            "fuse_mode_fomo": fused["fuse_mode_fomo"],
            "dir_conflict_crash": fused["dir_conflict_crash"],
            "dir_conflict_fomo": fused["dir_conflict_fomo"],
            "disagree_crash": fused["disagree_crash"],
            "disagree_fomo": fused["disagree_fomo"],
            "absdiff_crash": fused["absdiff_crash"],
            "absdiff_fomo": fused["absdiff_fomo"],
            "needs_review": fused["needs_review"],

            "llm_cache_hit": int(bool(llm.get("_cache_hit"))),
            "llm_evidence_crash": json.dumps((llm.get("evidence", {}) or {}).get("crash", []), ensure_ascii=False),
            "llm_evidence_fomo": json.dumps((llm.get("evidence", {}) or {}).get("fomo", []), ensure_ascii=False),
        }

        if keep_llm_raw:
            row["llm_raw"] = (llm.get("_raw", "") or "")[:4000]

        return idx, row, stats

    workers = max(1, int(workers))
    if workers == 1:
        for idx, a in enumerate(tqdm(deduped, desc=f"SchemeC scoring (model={model})")):
            _, row, stats = process_one(idx, a)
            rows_by_idx[idx] = row
            n_fail += stats["failed"]
            n_cache_hit += stats["cache_hit"]
            n_api_call += stats["api_call"]
    else:
        
        pbar = tqdm(total=len(deduped), desc=f"SchemeC scoring (model={model}, workers={workers})")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_one, idx, a) for idx, a in enumerate(deduped)]
            for fut in as_completed(futures):
                idx, row, stats = fut.result()
                rows_by_idx[idx] = row
                n_fail += stats["failed"]
                n_cache_hit += stats["cache_hit"]
                n_api_call += stats["api_call"]
                pbar.update(1)
        pbar.close()

    rows = [rows_by_idx[i] for i in range(len(deduped))]

    daily_fused = aggregate_daily(rows, crash_key="fused_crash_score", fomo_key="fused_fomo_score")
    daily_rule = aggregate_daily(rows, crash_key="rule_crash_score", fomo_key="rule_fomo_score")
    daily_llm = aggregate_daily(rows, crash_key="llm_crash_score", fomo_key="llm_fomo_score")

    ensure_dir_for_file(output_xlsx)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        daily_fused.to_excel(writer, index=False, sheet_name="daily_fused")
        daily_rule.to_excel(writer, index=False, sheet_name="daily_rule")
        daily_llm.to_excel(writer, index=False, sheet_name="daily_llm")
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="articles_scored")

    n_review = sum(1 for r in rows if r["needs_review"] == 1)

    print(f"Done. Wrote: {output_xlsx}")
    print(f"Loaded: {len(articles)} | Deduped: {len(deduped)} (key=publisher_norm+date+sha1)")
    if disable_llm:
        print("LLM disabled: YES")
    else:
        print(
            f"LLM model: {model} | API calls: {n_api_call} | Cache hits: {n_cache_hit} | "
            f"Failures: {n_fail} | force_llm={int(force_llm)} | thinking={thinking} | "
            f"max_chars={max_chars} | max_tokens={max_tokens} | workers={workers}"
        )
    print(f"Needs review: {n_review}")


# =============================================================================
# 7) CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    # Inputs (single or batch)
    ap.add_argument("--input_dir", type=str, default="", help="Batch directory containing PDFs/RTFs/CSVs (recursive).")
    ap.add_argument("--input_pdf", type=str, default="", help="Single PDF path (optional).")
    ap.add_argument("--input_rtf", type=str, default="", help="Single RTF path (optional).")
    ap.add_argument("--input_csv", type=str, default="", help="Already-extracted CSV path (optional).")

    # Extraction outputs
    ap.add_argument("--extract_csv", type=str, default="dowjones_news_combined.csv",
                    help="Where to write the combined extracted CSV (for batch or single). Use empty to skip writing.")
    ap.add_argument("--output_xlsx", type=str, required=True)

    # Extractor params
    ap.add_argument("--toc_scan_pages", type=int, default=DEFAULT_TOC_SCAN_PAGES)
    ap.add_argument("--toc_dot_threshold", type=int, default=DEFAULT_TOC_DOT_THRESHOLD)
    ap.add_argument("--min_title_len", type=int, default=DEFAULT_MIN_TITLE_LEN)
    ap.add_argument("--max_title_len", type=int, default=DEFAULT_MAX_TITLE_LEN)
    ap.add_argument("--max_page_gap", type=int, default=DEFAULT_MAX_PAGE_GAP,
                    help="Max page gap for PDF slicing; set -1 to disable.")

    # Scheme C params
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--weight_rule", type=float, default=0.3)
    ap.add_argument("--agree_tol", type=float, default=0.2)
    ap.add_argument("--keep_llm_raw", action="store_true")
    ap.add_argument("--cache_db", type=str, default="cache/llm_cache.sqlite")
    ap.add_argument("--disable_llm", action="store_true", help="Run rule-only (fused=rule).")
    ap.add_argument("--force_llm", action="store_true", help="Ignore cache reads; always call API (still writes cache).")

    # SPEED knobs (new)
    ap.add_argument("--max_chars", type=int, default=4000, help="Max chars fed to LLM per article (latency control).")
    ap.add_argument("--max_tokens", type=int, default=384, help="Max output tokens for LLM (JSON is small).")
    ap.add_argument("--thinking", type=str, default="disabled", choices=["disabled", "enabled"],
                    help="DeepSeek thinking control (disabled is faster).")
    ap.add_argument("--base_url", type=str, default="", help="Override OpenAI-compatible base URL (default: env or https://api.deepseek.com).")
    ap.add_argument("--no_response_format", action="store_true",
                    help="Disable response_format=json_object (compat fallback).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent workers for LLM calls (throughput boost; keep 1 if rate-limited).")

    args = ap.parse_args()

    if not (0.0 <= args.weight_rule <= 1.0):
        raise ValueError("weight_rule must be in [0,1].")
    if args.agree_tol < 0:
        raise ValueError("agree_tol must be >= 0.")
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be > 0.")

    max_page_gap = None if args.max_page_gap is None else (None if args.max_page_gap < 0 else args.max_page_gap)

    extracted_df = extract_inputs_to_df(
        input_pdf=args.input_pdf.strip(),
        input_rtf=args.input_rtf.strip(),
        input_csv=args.input_csv.strip(),
        input_dir=args.input_dir.strip(),
        toc_scan_pages=args.toc_scan_pages,
        toc_dot_threshold=args.toc_dot_threshold,
        min_title_len=args.min_title_len,
        max_title_len=args.max_title_len,
        max_page_gap=max_page_gap,
    )

    run_pipeline(
        extracted_df=extracted_df,
        extracted_csv_out=args.extract_csv.strip(),
        output_xlsx=args.output_xlsx,
        model=args.model,
        cache_db=(args.cache_db or ""),
        weight_rule=args.weight_rule,
        agree_tol=args.agree_tol,
        keep_llm_raw=args.keep_llm_raw,
        disable_llm=args.disable_llm,
        force_llm=args.force_llm,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        base_url=args.base_url.strip(),
        force_json_object=(not args.no_response_format),
        workers=args.workers,
    )

if __name__ == "__main__":
    main()
