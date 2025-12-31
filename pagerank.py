import numpy as np


class PageRank:
    def _init_uniform(self, n_nodes: int) -> np.ndarray:
        """
        Initializes the PageRank vector uniformly.

        Args:
            n_nodes (int): Number of nodes in the graph.

        Returns:
            np.ndarray: Uniformly initialized PageRank vector.
        """

        # TODO (PageRank)
        # 1. Initialize the PageRank vector uniformly.
        return np.ones(n_nodes) / n_nodes

    def _build_stochastic_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Initializes the stochastic matrix M.

        M[j, i] = 1 / out_degree(i) if (i, j) in E else 0

        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph.

        Returns:
            np.ndarray: Stochastic matrix M.
        """

        # TODO (PageRank)
        # 2. Initialize the stochastic matrix M.
        # M[j, i] = 1 / out_degree(i) if (i, j) in E else 0
        #
        # Hint:
        # A nested loop would suffice here,
        # but try optimizing the loops with vectorized matrix operations using numpy.
        # Not required, does not count towards your score, but encouraged ;-)

        M = np.zeros_like(adj_matrix, dtype=float) # 默认创建 int 类型的矩阵，会导致除法结果被截断成0
        out_degree = np.sum(adj_matrix, axis=1)
        for i in range(adj_matrix.shape[0]):
            if out_degree[i] > 0:
                M[:, i] = adj_matrix[i, :] / out_degree[i]
        return M

    def page_rank(
        self,
        adj_matrix: np.ndarray,
        beta: float = 0.8,
        max_iter: int = 40,
    ) -> np.ndarray:
        """
        PageRank algorithm. Compute the PageRank scores of the nodes in the given graph.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph.
            beta (float): Damping factor. Default is 0.8.
            max_iter (int): Maximum number of iterations. Default is 40.

        Returns:
            np.ndarray: 1D array of shape (num_nodes,) containing the PageRank scores.
        """

        num_nodes = adj_matrix.shape[0]

        ranks = self._init_uniform(num_nodes)
        M = self._build_stochastic_matrix(adj_matrix)

        # TODO (PageRank)
        # 3. Complete the power iteration
        # Iteratively update the ranks util max_iter is reached.
        for _ in range(max_iter):
            ranks = beta * M @ ranks + np.ones(num_nodes) * (1 - beta) / num_nodes

        return ranks
