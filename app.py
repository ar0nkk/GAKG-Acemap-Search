import streamlit as st
import concurrent.futures
import ast
from typing import List, Dict, Tuple, Optional

from main import (
    load_gakg,
    get_weighted_neighbors_pagerank,
    search_acemap,
    enhance_search_results,
    GAKG_PATH,
)
from ai_intent import AIIntentParser, RAGResearchAssistant

# Page configuration
st.set_page_config(
    page_title="GAKG Acemap Search Enhancement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_gakg_data():
    return load_gakg(GAKG_PATH)


@st.cache_resource
def get_ai_intent():
    return AIIntentParser()


@st.cache_resource
def get_rag_assistant():
    return RAGResearchAssistant()


GAKG_DF = get_gakg_data()

# Custom CSS
st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; margin: 0 auto; }
    .main { padding: 1rem; }
    .hint { color: #bbb; font-size: 0.9rem; font-style: italic; margin-top: 0.25rem; }
    .pill { display: inline-block; background: #222; color: #ddd; padding: 0.3rem 0.6rem; border-radius: 12px; font-size: 0.85rem; margin-right: 0.5rem; margin-bottom: 0.25rem; }
    .overlap { color: #86c5ff; font-size: 0.9rem; margin-top: 0.35rem; }
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
                api_results = future.result()
                for p in api_results:
                    pid = p.get("id") or p.get("title")
                    if pid and pid not in results_map:
                        results_map[pid] = p
            except Exception as exc:
                term = future_to_term[future]
                print(f"Search failed for term {term}: {exc}")

    merged_results = list(results_map.values())

    # 3) Enrich with graph overlap
    enhanced = enhance_search_results(merged_results, weighted_neighbors)

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

    return page_relevant, page_others, expansion_terms


def _map_sort_label(sort_token: Optional[str]) -> str:
    if sort_token == "citation":
        return "Most Cited"
    if sort_token == "date":
        return "Latest Published"
    return "Best Match"


def _update_query_state(raw_query: str):
    use_ai = st.session_state.get("use_ai", True)
    if use_ai:
        parsed = get_ai_intent().parse(raw_query)
        st.session_state.search_keyword = parsed.get("keyword") or raw_query
        sort_token = parsed.get("sort")
        st.session_state.intent_sort_token = sort_token or "relevance"
        st.session_state.intent_explanation = parsed.get("explanation", "")
    else:
        st.session_state.search_keyword = raw_query
        st.session_state.intent_sort_token = "relevance"
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
    st.session_state.intent_sort_token = "relevance"


# UI Layout
st.markdown("## 🔍 GAKG-based Acemap Search Enhancement", unsafe_allow_html=True)

# Session state defaults
if "page" not in st.session_state:
    st.session_state.page = 1
if "sort_option" not in st.session_state:
    st.session_state.sort_option = "Most Cited"
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
if "intent_sort_token" not in st.session_state:
    st.session_state.intent_sort_token = "relevance"
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
            ("Most Cited", "Latest Published"),
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
        st.session_state.intent_sort_token = "relevance"

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
    intent_sort_label = _map_sort_label(st.session_state.intent_sort_token)
    with st.container(border=True):
        st.markdown("#### 🧠 AI Search Assistant")
        st.markdown(f"- Core query: **{core_keyword or '(empty)'}**")
        exp = st.session_state.intent_explanation
        if exp:
            if exp == "fallback":
                st.markdown("- Reasoning: LLM returned no explanation (maybe API missing or parsing failed); using raw query and default sorting.")
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
