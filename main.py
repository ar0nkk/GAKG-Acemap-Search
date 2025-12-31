import pandas as pd
import requests
import numpy as np
from pagerank import PageRank
import os
import glob

# Configuration
GAKG_PATH = 'd:/Syncdisk/SJTU/DataMining/pilot/GAKG_Acemap_Search_Enhancement/data'
ACEMAP_API_URL = 'https://acemap.info/api/v1/work/search'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def load_gakg(data_dir):
    """Loads the GAKG dataset from the data directory."""
    print(f"Looking for GAKG data in {data_dir}...")
    
    # Try to find full chunks
    full_chunks = sorted(glob.glob(os.path.join(data_dir, "gakg_full_chunk_*.parquet")))
    
    if full_chunks:
        print(f"Found {len(full_chunks)} dataset chunks. Loading...")
        dfs = []
        for chunk_path in full_chunks:
            try:
                print(f"  Loading {os.path.basename(chunk_path)}...")
                dfs.append(pd.read_parquet(chunk_path))
            except Exception as e:
                print(f"  Error loading {chunk_path}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Successfully loaded GAKG dataset with {len(combined_df)} triples.")
            return combined_df
    
    print("No GAKG parquet files found. Please run download_data.py first.")
    return None

def get_weighted_neighbors_pagerank(gakg_df, keyword, top_k=20):
    """
    Finds neighbors of the keyword in the GAKG graph and ranks them using PageRank.
    Returns a dictionary of {neighbor: pagerank_score}.
    """
    keyword = keyword.lower()
    
    # 1. Extract a subgraph (e.g., 2-hop neighborhood)
    # For simplicity and performance, we'll start with 1-hop neighbors
    # and then find connections between them to build a local graph.
    
    # Find 1-hop neighbors
    subjects = gakg_df[gakg_df['subject'].str.lower() == keyword]['object'].tolist()
    objects = gakg_df[gakg_df['object'].str.lower() == keyword]['subject'].tolist()
    neighbors = set([n.lower() for n in subjects + objects])
    neighbors.add(keyword) # Include the keyword itself
    
    if not neighbors:
        return {}

    # Filter GAKG to include only edges where BOTH subject and object are in the neighbor set
    # This builds the induced subgraph on the neighborhood.
    # Note: This might be sparse. We could also include edges where at least one is in the set
    # but that might explode the graph size.
    
    # To make it efficient, we filter the DF.
    # We need case-insensitive matching.
    # Let's create a temporary lowercased DF for filtering if memory allows, or just iterate.
    # Given the subset size, let's try boolean indexing.
    
    # Optimization: Filter rows where subject OR object is in neighbors (to get a slightly larger context)
    # Or strictly induced subgraph: subject AND object in neighbors.
    # Let's go with induced subgraph for the "Concept Graph" of this topic.
    
    # We need to handle case sensitivity properly.
    # Let's assume the GAKG is mostly consistent or we lowercased it.
    # For this demo, we'll do a slow but correct filtering.
    
    mask_sub = gakg_df['subject'].str.lower().isin(neighbors)
    mask_obj = gakg_df['object'].str.lower().isin(neighbors)
    subgraph_df = gakg_df[mask_sub & mask_obj]
    
    if len(subgraph_df) == 0:
        # Fallback: if no internal edges, just return uniform weights
        return {n: 1.0 for n in neighbors}

    # 2. Build Adjacency Matrix
    # Map nodes to indices
    nodes = sorted(list(neighbors))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    for _, row in subgraph_df.iterrows():
        u = row['subject'].lower()
        v = row['object'].lower()
        if u in node_to_idx and v in node_to_idx:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            adj_matrix[idx_u, idx_v] = 1 # Unweighted or use count if multiple
            # If undirected, set both? GAKG is directed (subject -> object).
            # PageRank works on directed graphs.

    # 3. Run PageRank
    pr_solver = PageRank()
    # Handle dangling nodes (nodes with no outgoing edges) inside PageRank implementation
    # or ensure the matrix is stochastic. The provided PageRank implementation likely handles it
    # or expects us to. Let's check the implementation.
    # The provided implementation has _build_stochastic_matrix which handles 0 out-degree?
    # It sets column to 0. Standard PageRank handles dead ends by teleportation.
    
    pr_scores = pr_solver.page_rank(adj_matrix)
    
    # 4. Return dictionary
    result = {nodes[i]: pr_scores[i] for i in range(n_nodes)}
    
    # Normalize scores so the max is 1 (optional, for easier scoring interpretation)
    if result:
        max_score = max(result.values())
        if max_score > 0:
            for k in result:
                result[k] /= max_score
                
    return result

def get_neighbors(gakg_df, keyword):
    """
    Finds 1-hop neighbors of the keyword in the GAKG graph.
    Returns a set of neighbor nodes.
    """
    # This function is kept for reference but we will use the weighted version.
    return set(get_weighted_neighbors_pagerank(gakg_df, keyword).keys())

def search_acemap(keyword, page=1, size=10):
    """Searches Acemap for the keyword."""
    params = {
        'keyword': keyword,
        'order': 'desc',
        'page': page,
        'size': size
    }
    try:
        # Attempt to bypass system proxy which often causes issues on Windows
        proxies = {"http": None, "https": None}
        print(f"Requesting Acemap API: {ACEMAP_API_URL} with keyword='{keyword}'...")
        response = requests.get(ACEMAP_API_URL, params=params, headers=HEADERS, proxies=proxies, timeout=10)
        print(f"API Response Status: {response.status_code}")
        response.raise_for_status()
        results = response.json().get('results', [])
        print(f"API returned {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error searching Acemap: {e}")
        # Fallback: Try without proxy setting if the above fails (though unlikely to help if proxy is required)
        return []

def enhance_search_results(results, weighted_neighbors):
    """Builds enrichment scores by comparing paper keywords with the GAKG neighborhood."""
    print("Building enrichment scores from knowledge graph neighbors...")
    enhanced_results = []
    
    for paper in results:
        paper_keywords = []
        
        raw_keywords = paper.get('keywords', [])
        if raw_keywords:
            if isinstance(raw_keywords[0], dict):
                paper_keywords = [k.get('display_name', '').lower() for k in raw_keywords]
            else:
                paper_keywords = [str(k).lower() for k in raw_keywords]
        
        raw_concepts = paper.get('concepts', [])
        if raw_concepts:
            if isinstance(raw_concepts[0], dict):
                paper_keywords.extend([c.get('display_name', '').lower() for c in raw_concepts])
            else:
                paper_keywords.extend([str(c).lower() for c in raw_concepts])

        overlap = set(paper_keywords).intersection(weighted_neighbors.keys())
        score = sum(weighted_neighbors.get(k, 0.0) for k in overlap)

        paper['enhancement_score'] = score
        paper['overlapping_keywords'] = list(overlap)
        paper['has_graph_overlap'] = bool(overlap)
        enhanced_results.append(paper)

    return enhanced_results

def main():
    # 1. Load GAKG
    gakg_df = load_gakg(GAKG_PATH)
    if gakg_df is None:
        return

    # 2. User Input
    keyword = input("Enter search keyword (e.g., 'rock'): ").strip()
    if not keyword:
        keyword = 'rock' # Default
        print(f"Using default keyword: {keyword}")

    # --- Query Expansion Step ---
    print(f"\n[Query Expansion] Identifying related concepts for '{keyword}'...")
    # Run PageRank on the keyword's neighborhood to find top related concepts
    weighted_neighbors = get_weighted_neighbors_pagerank(gakg_df, keyword)
    
    # Sort neighbors by score, exclude the keyword itself
    sorted_neighbors = sorted(weighted_neighbors.items(), key=lambda x: x[1], reverse=True)
    expansion_terms = []
    for term, score in sorted_neighbors:
        if term.lower() != keyword.lower():
            expansion_terms.append(term)
        if len(expansion_terms) >= 3: # Pick top 3
            break
    
    if expansion_terms:
        print(f"Found related concepts: {', '.join(expansion_terms)}")
    else:
        print("No related concepts found for expansion.")

    # 3. Search Acemap (Original + Expanded)
    all_results_map = {} # id -> paper_dict

    # 3a. Search Original
    print(f"\nSearching Acemap for original keyword: '{keyword}'...")
    results = search_acemap(keyword, size=20)
    for p in results:
        all_results_map[p['id']] = p
        
    # 3b. Search Expanded
    for term in expansion_terms:
        print(f"Searching Acemap for related concept: '{term}'...")
        # Fetch fewer results for expanded terms to keep relevance high
        results = search_acemap(term, size=10) 
        for p in results:
            if p['id'] not in all_results_map:
                all_results_map[p['id']] = p
    
    merged_results = list(all_results_map.values())
    print(f"Total unique results retrieved: {len(merged_results)}")

    if not merged_results:
        print("No results found from Acemap.")
        return

    # 4. Enhance Results
    print("\nEnhancing results with GAKG...")
    enhanced_results = enhance_search_results(merged_results, weighted_neighbors)

    # 5. Display Results with Citation-aware Sorting
    citation_sort = lambda paper: (paper.get('cited_by_count', 0) or 0)
    relevant = sorted([p for p in enhanced_results if p['has_graph_overlap']], key=citation_sort, reverse=True)
    others = sorted([p for p in enhanced_results if not p['has_graph_overlap']], key=citation_sort, reverse=True)

    print("\n" + "="*50)
    print(f"Knowledge graph filtered results for '{keyword}'")
    print("="*50)

    if relevant:
        print("\nRelated papers (keywords linked to the GAKG neighborhood):")
        print("-" * 50)
        for i, paper in enumerate(relevant[:20], 1):
            print(f"{i}. [{paper.get('cited_by_count',0):,} cites] {paper.get('title')} ({paper.get('publication_year')}) | score {paper['enhancement_score']:.4f}")
            if paper['overlapping_keywords']:
                print(f"   Overlap: {', '.join(paper['overlapping_keywords'])}")
    else:
        print("No papers could be linked to the knowledge graph neighborhood.")

    if others:
        print("\nOther papers (sorted by citation count):")
        print("-" * 50)
        for i, paper in enumerate(others[:10], 1):
            print(f"{i}. [{paper.get('cited_by_count',0):,} cites] {paper.get('title')} ({paper.get('publication_year')})")

    print("\nDone.")


if __name__ == "__main__":
    main()
