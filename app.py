import os
import threading
import webbrowser
from typing import List, Dict, Tuple
from flask import Flask, request, render_template_string
from main import (
    load_gakg,
    get_weighted_neighbors_pagerank,
    search_acemap,
    enhance_search_results,
    GAKG_PATH,
)

# Load GAKG once at startup
GAKG_DF = load_gakg(GAKG_PATH)

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GAKG Acemap Search Enhancement</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f6fb; color: #222; }
    header { background: linear-gradient(135deg, #0f4c75, #3282b8); color: #fff; padding: 16px 20px; }
    header h1 { margin: 0; font-size: 20px; }
    main { max-width: 1100px; margin: 0 auto; padding: 20px; }
    form { margin: 16px 0; display: flex; gap: 12px; }
    input[type=text] { flex: 1; padding: 10px 12px; font-size: 16px; border: 1px solid #ccd1d9; border-radius: 6px; }
    button { padding: 10px 16px; font-size: 15px; background: #0f4c75; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #0c3d5f; }
    .hint { color: #555; font-size: 13px; margin-top: 4px; }
    .section { margin-top: 24px; }
    .section h2 { margin: 0 0 8px 0; font-size: 18px; color: #0f4c75; }
    .card { background: #fff; border: 1px solid #e3e6ed; border-radius: 8px; padding: 14px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .title { font-weight: 600; font-size: 15px; margin: 0 0 6px 0; }
    .meta { color: #555; font-size: 13px; margin-bottom: 6px; }
    .overlap { color: #0f4c75; font-size: 13px; }
    .tag { display: inline-block; background: #eef5fb; color: #0f4c75; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 6px; }
    .pill { display: inline-block; background: #f1f3f6; color: #444; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 6px; }
    a { color: #0f4c75; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .empty { color: #777; font-size: 14px; }
    .pagination { margin-top: 30px; display: flex; align-items: center; justify-content: center; gap: 10px; }
    .pagination a { padding: 8px 16px; border-radius: 6px; background: #0f4c75; color: #fff; text-decoration: none; font-weight: 500; }
    .pagination a:hover { background: #0c3d5f; }
    .pagination span { color: #555; font-size: 15px; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <h1>GAKG-based Acemap Search Enhancement</h1>
  </header>
  <main>
    <form method="get" action="/">
      <input type="text" name="q" placeholder="Search keyword (e.g., Plate Tectonics)" value="{{ query }}" />
      <button type="submit">Search</button>
    </form>
    <div class="hint">Workflow: Query expansion (GAKG) + API results merge + knowledge-graph overlap + citation-based ordering.</div>

    {% if expansion_terms %}
      <div class="section">
        <h2>Expanded terms</h2>
        {% for term in expansion_terms %}<span class="pill">{{ term }}</span>{% endfor %}
      </div>
    {% endif %}

    {% if relevant %}
      <div class="section">
        <h2>Relevant papers (graph-overlap, sorted by citations)</h2>
        {% for paper in relevant %}
          <div class=\"card\">
            <p class="title">{{ paper.rank_index }}. [{{ paper.cited_by_count|default(0) }} cites] <a href="{{ paper.link }}" target="_blank" rel="noopener">{{ paper.title }}</a></p>
            <div class="meta">Year: {{ paper.publication_year|default('N/A') }} | Score: {{ '%.4f'|format(paper.enhancement_score) }}</div>
            {% if paper.overlapping_keywords %}
              <div class="overlap">Overlap: {{ paper.overlapping_keywords|join(', ') }}</div>
            {% endif %}
          </div>
        {% endfor %}
      </div>
    {% else %}
      {% if query %}<div class="section empty">No papers could be linked to the GAKG neighborhood.</div>{% endif %}
    {% endif %}

    {% if others %}
      <div class="section">
        <h2>Other papers (sorted by citations)</h2>
        {% for paper in others %}
          <div class="card">
            <p class="title">{{ paper.rank_index }}. [{{ paper.cited_by_count|default(0) }} cites] <a href="{{ paper.link }}" target="_blank" rel="noopener">{{ paper.title }}</a></p>
            <div class="meta">Year: {{ paper.publication_year|default('N/A') }}</div>
          </div>
        {% endfor %}
      </div>
    {% endif %}

    {% if not relevant and not others and query %}
      <div class="section empty">No papers found.</div>
    {% endif %}

    {% if query %}
    <div class="pagination">
      {% if page > 1 %}
        <a href="?q={{ query|urlencode }}&page={{ page - 1 }}">← Previous</a>
      {% endif %}
      <span>Page {{ page }}</span>
      <a href="?q={{ query|urlencode }}&page={{ page + 1 }}">Next →</a>
    </div>
    {% endif %}
  </main>
</body>
</html>
"""


def build_paper_link(paper: Dict) -> str:
    # Prefer landing page, then DOI, then Acemap paper page.
    primary = paper.get("primary_location") or {}
    url = primary.get("landing_page_url")
    if not url and paper.get("doi"):
        url = f"https://doi.org/{paper['doi']}"
    if not url and paper.get("id"):
        url = f"https://acemap.info/paper/{paper['id']}"
    return url or "#"


def run_enhanced_pipeline(keyword: str, page: int) -> Tuple[List[Dict], List[Dict], List[str]]:
    if GAKG_DF is None:
        return [], [], []

    # 1) Neighborhood + expansion terms
    weighted_neighbors = get_weighted_neighbors_pagerank(GAKG_DF, keyword)
    sorted_neighbors = sorted(weighted_neighbors.items(), key=lambda x: x[1], reverse=True)
    expansion_terms: List[str] = []
    for term, _ in sorted_neighbors:
        if term.lower() != keyword.lower():
            expansion_terms.append(term)
        if len(expansion_terms) >= 3:
            break

    # 2) Fetch results: fetch a larger pool (e.g. 100) to allow global sorting
    # We ignore the 'page' param for the API call to ensure we get the top global results
    # and then paginate locally.
    results_map: Dict[str, Dict] = {}
    # Fetch more results for the main keyword to ensure high recall of top papers
    for term, size in [(keyword, 60)] + [(t, 20) for t in expansion_terms]:
      # Always fetch page 1 from API with larger size to get candidates
      api_results = search_acemap(term, page=1, size=size)
      for p in api_results:
        pid = p.get("id") or p.get("title")
        if pid and pid not in results_map:
          results_map[pid] = p
    merged_results = list(results_map.values())

    # 3) Enrich with graph overlap
    enhanced = enhance_search_results(merged_results, weighted_neighbors)

    # 4) Global Sort: graph-overlap first (by citations), then the rest (by citations)
    citation_sort = lambda paper: (paper.get('cited_by_count', 0) or 0)
    
    all_relevant = sorted([p for p in enhanced if p['has_graph_overlap']], key=citation_sort, reverse=True)
    all_others = sorted([p for p in enhanced if not p['has_graph_overlap']], key=citation_sort, reverse=True)
    
    # Combine for pagination
    full_list = all_relevant + all_others
    
    # Assign global rank index before slicing
    for idx, p in enumerate(full_list):
        p['rank_index'] = idx + 1
        p['link'] = build_paper_link(p)

    # 5) Paginate locally
    PER_PAGE = 20
    start_idx = (page - 1) * PER_PAGE
    end_idx = start_idx + PER_PAGE
    
    page_items = full_list[start_idx:end_idx]
    
    # Split back into relevant/others for display
    page_relevant = [p for p in page_items if p['has_graph_overlap']]
    page_others = [p for p in page_items if not p['has_graph_overlap']]

    return page_relevant, page_others, expansion_terms


@app.route("/", methods=["GET"])
def index():
  query = (request.args.get("q") or "").strip()
  raw_page = request.args.get("page", "1")
  try:
    page = max(1, int(raw_page))
  except ValueError:
    page = 1

  relevant: List[Dict] = []
  others: List[Dict] = []
  expansion_terms: List[str] = []

  if query:
    relevant, others, expansion_terms = run_enhanced_pipeline(query, page)

  return render_template_string(
    TEMPLATE,
    query=query,
    page=page,
    relevant=relevant,
    others=others,
    expansion_terms=expansion_terms,
  )


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))

  def _open_browser():
    # open the UI once the server is ready
    webbrowser.open(f"http://127.0.0.1:{port}/")

  # Prevent opening two windows when debug=True (reloader active)
  if not os.environ.get("WERKZEUG_RUN_MAIN"):
    threading.Timer(1.25, _open_browser).start()

  app.run(host="0.0.0.0", port=port, debug=True)
