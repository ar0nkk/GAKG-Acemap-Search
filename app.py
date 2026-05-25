import streamlit as st
import concurrent.futures
import ast
import math
from typing import List, Dict, Tuple, Optional

from main import (
    load_gakg,
    get_weighted_neighbors_pagerank,
    search_acemap,
    enhance_search_results,
    DATA_DIR,
)
from config import CITATION_WEIGHT_ALPHA
from agent import AIIntentParser, RAGResearchAssistant

# Page configuration
st.set_page_config(
    page_title="GAKG Nexus | Acemap Search",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_gakg_data():
    return load_gakg(DATA_DIR)


@st.cache_resource
def get_ai_intent():
    return AIIntentParser()


@st.cache_resource
def get_rag_assistant():
    return RAGResearchAssistant()


GAKG_DF = get_gakg_data()

# Custom CSS - Modern Dark Theme
st.markdown(
    """
    <style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1a1c24 0%, #0f1016 90%);
        color: #e0e0e0;
    }

    /* Constraints layout width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: auto;
    }

    /* Titles & Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Hide "Press Enter to submit form" hint */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        padding-bottom: 0.5rem;
    }

    /* Custom Cards */
    .paper-card {
        background: rgba(30, 34, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
        backdrop-filter: blur(10px);
    }
    
    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        background: rgba(40, 44, 55, 0.8);
        border-color: rgba(79, 172, 254, 0.4);
    }

    .paper-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #64b5f6;
        text-decoration: none;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .paper-title:hover {
        text-decoration: underline;
        color: #9be7ff;
    }

    .paper-meta {
        font-size: 0.85rem;
        color: #a0a0b0;
        margin-bottom: 0.8rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .paper-abstract {
        font-size: 0.9rem;
        color: #cfcfde;
        line-height: 1.5;
        margin-top: 0.5rem;
        border-left: 3px solid #4facfe;
        padding-left: 0.8rem;
    }

    /* Pills / Tags */
    .pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .pill {
        background: linear-gradient(135deg, #2b3242 0%, #1e222d 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #00f2fe;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .overlap-tag {
        background: rgba(79, 172, 254, 0.15);
        color: #86c5ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        border: 1px solid rgba(79, 172, 254, 0.3);
        margin-top: 0.5rem;
        display: inline-block;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #111319;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Inputs */
    .stTextInput input, .stSelectbox > div > div {
        background-color: #1e222d !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        font-weight: 700;
        border: none;
        transition: opacity 0.2s;
    }
    .stButton button:hover {
        opacity: 0.9;
        color: #000;
    }
    
    /* Metrics */
    .score-badge {
        background-color: rgba(0, 255, 128, 0.1);
        color: #00ff80;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def build_paper_link(paper: Dict) -> str:
    """Build a clickable link for a paper."""
    primary = paper.get("primary_location") or {}
    url = primary.get("landing_page_url")
    if not url and paper.get("doi"):
        url = f"https://doi.org/{paper['doi']}"
    if not url and paper.get("id"):
        url = f"https://acemap.info/paper/{paper['id']}"
    return url or "#"


def render_paper_card(paper: Dict, is_overlap: bool = False):
    """Render a beautiful card for a paper result using HTML."""
    link = build_paper_link(paper)
    title = paper.get('title', 'Untitled')
    year = paper.get("publication_year", "N/A")
    score = paper.get("enhancement_score", 0)
    cited_by = paper.get("cited_by_count", 0) or 0
    rank = paper.get("rank_index", "?")
    
    author_list = paper.get("author_names", [])[:3]
    if len(paper.get("author_names", [])) > 3:
        author_list.append("et al.")
    authors = ", ".join(author_list)
    
    inst_list = paper.get("affiliation_names", [])[:1]
    insts = ", ".join(inst_list)
    
    abstract = get_abstract_preview(paper)
    if abstract:
        # Strip the markdown prefix if present in the preview function
        abstract = abstract.replace("**Abstract:**", "").strip()

    overlap_html = ""
    if paper.get("overlapping_keywords"):
        overlap_str = ", ".join(paper["overlapping_keywords"])
        overlap_html = f'<div class="overlap-tag">🔗 Overlap: {overlap_str}</div>'

    score_badge = f'<span class="score-badge">Score: {score:.4f}</span>' if is_overlap else ""

    html = f"""
    <div class="paper-card">
        <a href="{link}" target="_blank" class="paper-title">{rank}. {title}</a>
        <div class="paper-meta">
            <span class="meta-tag">📅 {year}</span>
            <span class="meta-tag">⭐ {cited_by} Citations</span>
            {f'<span class="meta-tag">{score_badge}</span>' if score_badge else ''}
        </div>
        <div class="paper-meta" style="color: #889;">
            {f'<span class="meta-tag">👤 {authors}</span>' if authors else ''}
            {f'<span class="meta-tag">🏛️ {insts}</span>' if insts else ''}
        </div>
        {f'<div class="paper-abstract">{abstract}</div>' if abstract else ''}
        {overlap_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def extract_authors_affiliations(paper: Dict) -> Tuple[List[str], List[str]]:

    """Extract author and affiliation names from varied schema fields."""
    authors: List[str] = []
    insts: List[str] = []

    def _pick_name(obj: Dict) -> Optional[str]:
        return obj.get("display_name") or obj.get("name") or obj.get("author") or obj.get("fullname")

    def _clean_name(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        if "{" in s or "}" in s:
            return None
        if s.lower().startswith("id:"):
            return None
        # Strip surrounding quotes/brackets that come from list-like strings
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1].strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].strip()
        return s or None

    auth_list = paper.get("authorships") or paper.get("authors") or []
    if isinstance(auth_list, list):
        for a in auth_list:
            if isinstance(a, dict):
                name = _clean_name(_pick_name(a))
                # OpenAlex-style: nested author dict inside authorships
                if not name and isinstance(a.get("author"), dict):
                    name = _clean_name(_pick_name(a["author"]))
                if name:
                    authors.append(name)
                for inst in a.get("institutions") or a.get("affiliations") or []:
                    if isinstance(inst, dict):
                        iname = _clean_name(_pick_name(inst))
                        if iname:
                            insts.append(iname)
            elif isinstance(a, str):
                s = a.strip()
                parsed = None
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = ast.literal_eval(s)
                    except Exception:
                        parsed = None
                if isinstance(parsed, list):
                    for item in parsed:
                        name = _clean_name(item)
                        if name:
                            authors.append(name)
                else:
                    name = _clean_name(s)
                    if name:
                        authors.append(name)
            # ignore non-str, non-dict entries

    # Fallbacks
    if not authors and isinstance(paper.get("author"), str):
        val = _clean_name(paper.get("author"))
        if val:
            authors.append(val)
    if not authors and isinstance(paper.get("author"), list):
        for a in paper.get("author", []):
            name = _clean_name(_pick_name(a) if isinstance(a, dict) else a)
            if name:
                authors.append(name)
    if paper.get("institutions") and isinstance(paper["institutions"], list):
        for inst in paper["institutions"]:
            if isinstance(inst, dict):
                iname = _clean_name(_pick_name(inst))
                if iname:
                    insts.append(iname)
            elif isinstance(inst, str):
                s = inst.strip()
                parsed = None
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = ast.literal_eval(s)
                    except Exception:
                        parsed = None
                if isinstance(parsed, list):
                    for item in parsed:
                        iname = _clean_name(item)
                        if iname:
                            insts.append(iname)
                else:
                    iname = _clean_name(s)
                    if iname:
                        insts.append(iname)

    # Deduplicate while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return _dedup(authors), _dedup(insts)


def get_abstract_preview(paper: Dict, max_chars: int = 260) -> str:
    """Return a short abstract preview if available."""
    abstract = paper.get("abstract") or paper.get("abstract_text")
    if isinstance(abstract, str) and abstract.strip():
        text = abstract.strip()
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    inv = paper.get("abstract_inverted_index")
    if isinstance(inv, dict) and inv:
        # Reconstruct abstract from inverted index (OpenAlex-style)
        positions = []
        for word, idxs in inv.items():
            for i in idxs:
                positions.append((i, word))
        if positions:
            max_pos = max(i for i, _ in positions)
            tokens = [""] * (max_pos + 1)
            for i, w in positions:
                tokens[i] = w
            text = " ".join(t for t in tokens if t)
            text = text.strip()
            return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
    return ""


def run_enhanced_pipeline(
    keyword: str,
    page: int,
    sort_option: str,
    author_filter: str = "",
    affiliation_filter: str = "",
    neighbor_threshold: float = 0.0,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Run the enhanced search pipeline with sorting. Keyword is already LLM-parsed."""
    if GAKG_DF is None:
        return [], [], []

    api_sort = None
    if sort_option == "Most Cited":
        api_sort = "cited_by_count"
    elif sort_option == "Latest Published":
        api_sort = "publication_date"
    # For "Most Relevant" and "Default", we use default API relevance (None)

    # 1) Neighborhood + expansion terms from GAKG (keyword already parsed by LLM)
    core_keyword = keyword
    weighted_neighbors = get_weighted_neighbors_pagerank(GAKG_DF, core_keyword)
    # Apply threshold to expansion candidates
    sorted_neighbors = [
        (term, score)
        for term, score in sorted(weighted_neighbors.items(), key=lambda x: x[1], reverse=True)
        if score >= neighbor_threshold
    ]
def _check_relevance(paper: Dict, keyword: str) -> bool:
    """Check if the paper content is relevant to the keyword (Title/Abstract/Keywords)."""
    kw_lower = keyword.lower()
    
    # 1. Title
    if kw_lower in (paper.get("title") or "").lower():
        return True
        
    # 2. Keywords
    p_kws = []
    raw_kws = paper.get("keywords") or []
    if raw_kws:
        if isinstance(raw_kws[0], dict):
            p_kws = [k.get('display_name', '').lower() for k in raw_kws]
        else:
            p_kws = [str(k).lower() for k in raw_kws]
    if any(kw_lower in k for k in p_kws):
        return True
        
    # 3. Abstract (Handle Inverted Index if possible, simpler to check raw text match if available)
    # Acemap abstract is often an inverted index. Reconstructing it is expensive.
    # We check if the keys contain the words of the keyword.
    ab = paper.get("abstract")
    if isinstance(ab, str):
        if kw_lower in ab.lower():
            return True
    elif isinstance(ab, dict) and "inverted_index" in ab:
        # Check if all tokens of the keyword appear in the index keys
        # Simple heuristic: exact match of the phrase is hard in inverted index without position.
        # So we check if *all* words in the keyword are present as keys.
        tokens = kw_lower.split()
        index_keys = set(k.lower() for k in ab["inverted_index"].keys())
        if all(t in index_keys for t in tokens):
            return True

    return False

def run_enhanced_pipeline(
    keyword: str,
    page: int,
    sort_option: str,
    author_filter: str = "",
    affiliation_filter: str = "",
    neighbor_threshold: float = 0.0,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Run the enhanced search pipeline with sorting. Keyword is already LLM-parsed."""
    if GAKG_DF is None:
        return [], [], []

    api_sort = None
    if sort_option == "Most Cited":
        api_sort = "cited_by_count"
    elif sort_option == "Latest Published":
        api_sort = "publication_date"
    # For "Most Relevant" and "Relevance & Impact", we use default API relevance (None)

    # 1) Neighborhood + expansion terms from GAKG (keyword already parsed by LLM)
    core_keyword = keyword
    weighted_neighbors = get_weighted_neighbors_pagerank(GAKG_DF, core_keyword)
    # Apply threshold to expansion candidates
    sorted_neighbors = [
        (term, score)
        for term, score in sorted(weighted_neighbors.items(), key=lambda x: x[1], reverse=True)
        if score >= neighbor_threshold
    ]
    # If the threshold is maxed (~1), skip expansion terms
    if neighbor_threshold >= 0.999:
        sorted_neighbors = []
    expansion_terms: List[str] = []
    for term, _ in sorted_neighbors:
        if term.lower() != core_keyword.lower():
            expansion_terms.append(term)
        if len(expansion_terms) >= 3:
            break

    # 2) Fetch results (parallel)
    results_map: Dict[str, Dict] = {}
    # Increase fetch sizes to broaden recall before reranking
    queries_to_run = [(core_keyword, 100)] + [(t, 40) for t in expansion_terms]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_term = {
            executor.submit(search_acemap, term, page=1, size=size, sort=api_sort): term
            for term, size in queries_to_run
        }
        for future in concurrent.futures.as_completed(future_to_term):
            try:
                term = future_to_term[future]
                api_results = future.result()
                is_core = (term.lower() == core_keyword.lower())
                
                for p in api_results:
                    pid = p.get("id") or p.get("title")
                    if not pid: continue
                    
                    if pid not in results_map:
                        p['_found_by_core'] = is_core
                        results_map[pid] = p
                    else:
                        if is_core:
                            results_map[pid]['_found_by_core'] = True
                            
            except Exception as exc:
                term = future_to_term[future]
                print(f"Search failed for term {term}: {exc}")

    # Deduplication & Relevance Filtering
    final_candidates = []
    for p in results_map.values():
        # Keep if found by core keyword
        if p.get('_found_by_core', False):
            final_candidates.append(p)
            continue
            
        # If found ONLY by expansion, STRICTLY check relevance to core keyword
        # "Discard if completely irrelevant to query word"
        if _check_relevance(p, core_keyword):
            final_candidates.append(p)
        # else: discard

    # 3) Enrich with graph overlap
    enhanced = enhance_search_results(final_candidates, weighted_neighbors)

    # 4) Global Sort (by user-selected criterion)
    def _date_key(p: Dict) -> int:
        date_str = p.get("publication_date") or ""
        if date_str:
            try:
                # Expect YYYY-MM-DD; fallback to year if parsing fails
                return int(date_str.replace("-", "")[:8])
            except Exception:
                pass
        year = p.get("publication_year")
        try:
            return int(year)
        except Exception:
            return 0

    def get_secondary_sort(p):
        if sort_option == "Latest Published":
            return _date_key(p)
        elif sort_option == "Most Relevant":
            # Sort by GAKG score first, then citations
            return (p.get("enhancement_score", 0), p.get("cited_by_count", 0) or 0)
        elif sort_option == "Default":
            # Integrated Score: GAKG Score + alpha * log(Citations)
            # This balances topical relevance with paper impact
            score = p.get("enhancement_score", 0)
            cited = p.get("cited_by_count", 0) or 0
            return score + CITATION_WEIGHT_ALPHA * math.log(cited + 1)
        
        # Default fallback (Most Cited)
        return p.get("cited_by_count", 0) or 0

    # Enrich with author/affiliation metadata
    for p in enhanced:
        authors, insts = extract_authors_affiliations(p)
        p["author_names"] = authors
        p["affiliation_names"] = insts

    # Apply author / affiliation filters (case-insensitive substring match)
    a_f = (author_filter or "").strip().lower()
    inst_f = (affiliation_filter or "").strip().lower()

    def _match_filters(p: Dict) -> bool:
        if a_f:
            names = [n.lower() for n in p.get("author_names", [])]
            if not any(a_f in n for n in names):
                return False
        if inst_f:
            insts = [n.lower() for n in p.get("affiliation_names", [])]
            if not any(inst_f in n for n in insts):
                return False
        return True

    filtered = [p for p in enhanced if _match_filters(p)]

    # Sort purely by the selected option (date or citations)
    enhanced_sorted = sorted(filtered, key=get_secondary_sort, reverse=True)

    full_list = enhanced_sorted
    for idx, p in enumerate(full_list):
        p["rank_index"] = idx + 1
        p["link"] = build_paper_link(p)

    PER_PAGE = 20
    start_idx = (page - 1) * PER_PAGE
    end_idx = start_idx + PER_PAGE
    page_items = full_list[start_idx:end_idx]
    page_relevant = [p for p in page_items if p["has_graph_overlap"]]
    page_others = [p for p in page_items if not p["has_graph_overlap"]]

    # Recalculate expansion terms based on actual overlaps found in the results
    actual_expansion_terms = set()
    for p in enhanced_sorted:
        if 'overlapping_keywords' in p:
            actual_expansion_terms.update(p['overlapping_keywords'])
    
    new_expansion_terms = []
    for term in actual_expansion_terms:
         if term.lower() != core_keyword.lower():
             new_expansion_terms.append(term)
    
    # Sort by original graph weight
    new_expansion_terms.sort(key=lambda t: weighted_neighbors.get(t, 0), reverse=True)

    return page_relevant, page_others, new_expansion_terms


def _update_query_state(raw_query: str):
    use_ai = st.session_state.get("use_ai", True)
    if use_ai:
        parsed = get_ai_intent().parse(raw_query)
        st.session_state.search_keyword = parsed.get("keyword") or raw_query
        st.session_state.intent_explanation = parsed.get("explanation", "")
    else:
        st.session_state.search_keyword = raw_query
        st.session_state.intent_explanation = "AI disabled"
    st.session_state.query_raw = raw_query
    st.session_state.page = 1
    st.session_state.rag_answer = None


def _on_query_change():
    """Sync parsed fields when the user edits the search box to avoid stale concat."""
    current = (st.session_state.query_input or "").strip()
    st.session_state.search_keyword = current
    st.session_state.query_raw = current
    st.session_state.rag_answer = None
    st.session_state.intent_explanation = ""


# UI Layout
st.markdown("## 🔍 GAKG-based Acemap Search Enhancement", unsafe_allow_html=True)

# Session state defaults
if "page" not in st.session_state:
    st.session_state.page = 1
if "sort_option" not in st.session_state:
    st.session_state.sort_option = "Default"
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "query_raw" not in st.session_state:
    st.session_state.query_raw = ""
if "search_keyword" not in st.session_state:
    st.session_state.search_keyword = ""
if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = None
if "intent_explanation" not in st.session_state:
    st.session_state.intent_explanation = ""
if "author_filter" not in st.session_state:
    st.session_state.author_filter = ""
if "affiliation_filter" not in st.session_state:
    st.session_state.affiliation_filter = ""
if "neighbor_threshold" not in st.session_state:
    st.session_state.neighbor_threshold = 0.0
if "use_ai" not in st.session_state:
    st.session_state.use_ai = True


# Search bar (Enter submits via form)
with st.form("search_form"):
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    with col1:
        st.text_input(
            "Search keyword",
            key="query_input",
            placeholder="e.g., plate tectonics",
            label_visibility="collapsed",
        )
    with col2:
        st.selectbox(
            "Sort",
            ("Default", "Most Relevant", "Most Cited", "Latest Published"),
            key="sort_option",
        )
    with col3:
        st.checkbox(
            "AI Search",
            key="use_ai",
            help="Off: use raw query without LLM parsing/assistant",
        )
    with col4:
        search_button = st.form_submit_button("Search", use_container_width=True)

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        st.text_input(
            "Author (optional)",
            key="author_filter",
            placeholder="e.g., John Smith",
        )
    with col5:
        st.text_input(
            "Institution (optional)",
            key="affiliation_filter",
            placeholder="e.g., MIT",
        )
    with col6:
        st.slider(
            "Expansion limitation",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="neighbor_threshold",
            help="Lower this if few results come back",
        )

# Display hint
st.markdown(
    '<p class="hint">💡 Workflow: LLM intent parsing + KG expansion + API merge + knowledge-graph overlap + citation/date sorting.</p>',
    unsafe_allow_html=True,
)

# Handle search
if search_button:
    raw_query = (st.session_state.query_input or "").strip()
    if raw_query:
        if st.session_state.get("use_ai", True):
            with st.spinner("Understanding query with LLM..."):
                _update_query_state(raw_query)
        else:
            _update_query_state(raw_query)
    else:
        st.session_state.query_raw = ""
        st.session_state.search_keyword = ""
        st.session_state.rag_answer = None
        st.session_state.page = 1
        st.session_state.intent_explanation = ""

query_raw = st.session_state.query_raw
core_keyword = st.session_state.search_keyword or st.session_state.query_input
page = st.session_state.page
sort_opt = st.session_state.sort_option
author_filter = st.session_state.author_filter
affiliation_filter = st.session_state.affiliation_filter
neighbor_threshold = st.session_state.neighbor_threshold

if core_keyword:
    with st.spinner(f"Searching and processing results... (Sort: {sort_opt})"):
        relevant, others, expansion_terms = run_enhanced_pipeline(
            core_keyword,
            page,
            sort_opt,
            author_filter=author_filter,
            affiliation_filter=affiliation_filter,
            neighbor_threshold=neighbor_threshold,
        )

    # AI Search Assistant: show LLM understanding + generate summary/suggestions
    with st.container(border=True):
        st.markdown("#### 🧠 AI Search Assistant")
        st.markdown(f"- Core query: **{core_keyword or '(empty)'}**")
        exp = st.session_state.intent_explanation
        if exp:
            if exp == "fallback":
                st.markdown("- Reasoning: LLM returned no explanation (maybe API missing or parsing failed); using raw query.")
            else:
                st.markdown(f"- Reasoning: {exp}")

        use_ai = st.session_state.get("use_ai", True)
        rag_disabled = (not use_ai) or not (relevant or others)
        if st.button("Generate search tips & summary", key="rag_generate", disabled=rag_disabled):
            with st.spinner("AI assistant is generating..."):
                st.session_state.rag_answer = get_rag_assistant().answer(
                    user_query=query_raw or core_keyword,
                    keyword=core_keyword,
                    expansion_terms=expansion_terms,
                    relevant=relevant,
                    others=others,
                    language=query_raw.strip()[:1] if query_raw else None,
                )
        if not use_ai:
            st.caption("AI assistant disabled (toggle off).")
        elif rag_disabled:
            st.caption("Search results are needed before generating an answer.")
        if st.session_state.rag_answer:
            st.markdown(st.session_state.rag_answer)

    # Expansion terms
    if expansion_terms:
        st.markdown("### 📚 Expansion Terms")
        term_html = " ".join([f'<span class="pill">{term}</span>' for term in expansion_terms])
        st.markdown(term_html, unsafe_allow_html=True)

    # Relevant papers
    if relevant:
        st.markdown(f"### ✨ Graph-overlap results (sorted by: {sort_opt})")
        for paper in relevant:
            with st.container(border=True):
                cited_by = paper.get("cited_by_count", 0) or 0
                title_html = f"**{paper.get('rank_index', '?')}. [{cited_by} cites] [{paper.get('title', 'Untitled')}]({paper.get('link', '#')})**"
                st.markdown(title_html)

                year = paper.get("publication_year", "N/A")
                score = paper.get("enhancement_score", 0)
                st.caption(f"Year: {year} | Score: {score:.4f}")
                authors_line = ", ".join(paper.get("author_names", [])[:3])
                insts_line = ", ".join(paper.get("affiliation_names", [])[:2])
                if authors_line:
                    st.caption(f"Authors: {authors_line}")
                if insts_line:
                    st.caption(f"Institutions: {insts_line}")
                abstract_preview = get_abstract_preview(paper)
                if abstract_preview:
                    st.markdown(abstract_preview)

                if paper.get("overlapping_keywords"):
                    overlap_str = ", ".join(paper["overlapping_keywords"])
                    st.markdown(f'<p class="overlap">Overlap: {overlap_str}</p>', unsafe_allow_html=True)

    # Other papers
    if others:
        st.markdown(f"#### 📄 No graph overlap for the results below (sorted by: {sort_opt})")
        for paper in others:
            with st.container(border=True):
                cited_by = paper.get("cited_by_count", 0) or 0
                title_html = f"**{paper.get('rank_index', '?')}. [{cited_by} cites] [{paper.get('title', 'Untitled')}]({paper.get('link', '#')})**"
                st.markdown(title_html)

                year = paper.get("publication_year", "N/A")
                st.caption(f"Year: {year}")
                authors_line = ", ".join(paper.get("author_names", [])[:3])
                insts_line = ", ".join(paper.get("affiliation_names", [])[:2])
                if authors_line:
                    st.caption(f"Authors: {authors_line}")
                if insts_line:
                    st.caption(f"Institutions: {insts_line}")
                abstract_preview = get_abstract_preview(paper)
                if abstract_preview:
                    st.markdown(abstract_preview)

    if not relevant and not others:
        st.info("❌ No papers could be linked to the GAKG neighborhood.")

    # Pagination
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if page > 1 and st.button("← Previous Page"):
            st.session_state.page -= 1
            st.rerun()
    with col2:
        st.markdown(
            f"<p style='text-align: center; font-weight: bold;'>Page {page}</p>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button("Next Page →"):
            st.session_state.page += 1
            st.rerun()
else:
    st.info("👋 Enter a search keyword to get started!")
