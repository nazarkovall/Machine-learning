import networkx as nx
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

class GraphML:
    def __init__(self, adjacency_matrix=None, edge_list=None):
        if adjacency_matrix is not None:
            self.graph = nx.from_numpy_array(np.array(adjacency_matrix))
        elif edge_list is not None:
            self.graph = nx.Graph()
            self.graph.add_edges_from(edge_list)
        else:
            self.graph = nx.Graph()

    def bfs(self, start):
        return list(nx.bfs_edges(self.graph, start))

    def dfs(self, start):
        return list(nx.dfs_edges(self.graph, start))

    def shortest_path(self, source, target):
        return nx.shortest_path(self.graph, source, target)

    def all_paths_length(self, source, target, length):
        return [path for path in nx.all_simple_paths(self.graph, source, target) if len(path) - 1 == length]

    def find_cycles(self):
        return list(nx.cycle_basis(self.graph))

    def degree_matrix(self):
        degrees = dict(self.graph.degree())
        return np.diag([degrees[node] for node in self.graph.nodes()])

    def local_overlap_measures(self):
        overlap = {}
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v:
                    neighbors_u = set(self.graph.neighbors(u))
                    neighbors_v = set(self.graph.neighbors(v))
                    common_neighbors = len(neighbors_u & neighbors_v)
                    total_neighbors = len(neighbors_u | neighbors_v)
                    overlap[(u, v)] = common_neighbors / total_neighbors if total_neighbors > 0 else 0
        return overlap

    def adjacency_eigenvalues_vectors(self):
        adj_matrix = nx.to_numpy_array(self.graph)
        eigenvalues, eigenvectors = eigh(adj_matrix)
        return eigenvalues, eigenvectors

    def laplacian_matrix(self, normalized=False):
        if normalized:
            return nx.normalized_laplacian_matrix(self.graph).toarray()
        return nx.laplacian_matrix(self.graph).toarray()

    def spectral_clustering(self, k=2):
        L = self.laplacian_matrix(normalized=True)
        _, eigenvectors = eigh(L)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(eigenvectors[:, :k])
        return labels
