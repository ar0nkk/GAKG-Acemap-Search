from agent import AIIntentParser
from config import DATA_DIR, MODEL_NAME, OPENAI_API_KEY
from main import get_weighted_neighbors_pagerank, load_gakg, search_acemap


def main() -> None:
    print(f"model={MODEL_NAME}")
    print(f"api_key_configured={bool(OPENAI_API_KEY)}")

    gakg_df = load_gakg(DATA_DIR)
    if gakg_df is None:
        raise SystemExit("failed_to_load_gakg")
    print(f"gakg_edges={len(gakg_df)}")

    keyword = "plate tectonics"
    neighbors = get_weighted_neighbors_pagerank(gakg_df, keyword, top_k=8)
    top_neighbors = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)[:8]
    print("top_neighbors=" + ", ".join(f"{term}:{score:.3f}" for term, score in top_neighbors))

    papers = search_acemap(keyword, size=3)
    print(f"acemap_results={len(papers)}")
    if papers:
        print("first_title=" + str(papers[0].get("title", ""))[:120])

    parsed = AIIntentParser().parse("latest influential papers about plate tectonics")
    print(f"parsed_keyword={parsed.get('keyword')}")
    print(f"parsed_explanation={parsed.get('explanation')}")


if __name__ == "__main__":
    main()
