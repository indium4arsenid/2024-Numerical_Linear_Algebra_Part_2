import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import logging
import sys
import os

# ========================
# Configuration Variables
# ========================
NUM_WEBSITES = 20          # Number of websites (n)
BETA = 1.0                 # Baseline probability of linking to less connected websites
DAMPING_FACTOR = 0.85      # Damping factor (alpha)
RANDOM_SEED = 42           # Seed for reproducibility
TOLERANCE = 1e-8           # Tolerance for power method convergence
MAX_ITERATIONS = 1000      # Maximum iterations for power method

# Visualization Flags
ENABLE_VERBOSE = False         # Enable verbose output
ENABLE_VISUALIZE = True        # Plot the PageRank distribution
ENABLE_PLOT_CONVERGENCE = True # Plot the convergence of the power method
ENABLE_PLOT_GRAPH = True       # Visualize the web graph with PageRank

# Matrix Type
USE_SPARSE_MATRICES = False    # Use sparse matrices for adjacency and Google matrices

# Save Directory for Plots
SAVE_DIRECTORY = "plots_pref"        # Directory to save the plots (set to None to disable saving)

# ========================
# Configure Logging
# ========================
logging_level = logging.INFO if ENABLE_VERBOSE else logging.WARNING
logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')

# ========================
# Function Definitions
# ========================
def generate_preferential_adjacency(n, beta, seed=None, sparse=False):
    """
    Generates an adjacency matrix using the preferential attachment model.

    Parameters:
    - n (int): Number of websites.
    - beta (float): Baseline likelyhood of a new website linking to less connected websites.
    - expected_out_degree (float): Desired expected number of outgoing links per website.
    - seed (int, optional): Seed for the random number generator.
    - sparse (bool): If True, returns a sparse matrix.

    Returns:
    - numpy.ndarray or scipy.sparse.csr_matrix: An n x n adjacency matrix.
    """
    if seed is not None:
        np.random.seed(seed)
        logging.debug(f"Random seed set to {seed}")

    # Initialize adjacency matrix
    A = np.zeros((n, n), dtype=int)
    p = np.zeros((n))

    for k in range(1, n):  # Websites are 0-indexed; website 1 is index 0
        j_indices = np.arange(k)  # Websites 0 to k-1
        print('j_indices',j_indices)
        norm_sum = k * beta + np.count_nonzero(A) # Number of total links + scaled baseline factor
        print('norm_sum',norm_sum)
        for j in j_indices:
            d_j = np.count_nonzero(A[:, j]) # Number of existing links to website j
            p[j] = (d_j + beta) / norm_sum

        # Generate random links
        links = np.random.rand(k) < p[:k]
        A[k, j_indices] = links.astype(int)
        logging.debug(f"Website {k+1}: p_k={p[:k]}, Links={A[k, j_indices]}")

    if sparse:
        A = csr_matrix(A)
        logging.debug("Converted adjacency matrix to sparse format.")
    return A

def adjacency_to_stochastic(A):
    """
    Converts an adjacency matrix to a row-stochastic matrix.

    Parameters:
    - A (numpy.ndarray or scipy.sparse.csr_matrix): Adjacency matrix.

    Returns:
    - numpy.ndarray or scipy.sparse.csr_matrix: Row-stochastic matrix P.
    """
    if isinstance(A, csr_matrix):
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums_with_no_zero = np.where(row_sums == 0, 1, row_sums)
        # Normalize rows
        P = A.multiply(1 / row_sums_with_no_zero[:, np.newaxis])
        # Handle dangling nodes
        dangling_nodes = (row_sums == 0)
        if dangling_nodes.any():
            n = A.shape[0]
            dangling_matrix = csr_matrix((np.ones(n), (np.arange(n), np.arange(n))))
            # Convert P to dense format to handle dangling nodes
            P_dense = P.toarray()
            P_dense[dangling_nodes] = 1.0 / n
            P = csr_matrix(P_dense)
            logging.debug("Handled dangling nodes in sparse matrix.")
        return P
    else:
        row_sums = A.sum(axis=1)
        row_sums_with_no_zero = np.where(row_sums == 0, 1, row_sums)
        P = A / row_sums_with_no_zero[:, np.newaxis]
        # Handle dangling nodes by assigning uniform probabilities
        dangling_nodes = (row_sums == 0)
        P[dangling_nodes] = 1.0 / A.shape[0]
        logging.debug("Converted adjacency matrix to row-stochastic matrix.")
        return P

def google_matrix(P, alpha=0.85, sparse=False):
    """
    Constructs the Google matrix.

    Parameters:
    - P (numpy.ndarray or scipy.sparse.csr_matrix): Row-stochastic matrix.
    - alpha (float): Damping factor.
    - sparse (bool): If True, returns a sparse Google matrix.

    Returns:
    - numpy.ndarray or scipy.sparse.csr_matrix: Google matrix G.
    """
    n = P.shape[0]
    if sparse and isinstance(P, csr_matrix):
        E = csr_matrix(np.ones((n, n)) / n)
        G = alpha * P + (1 - alpha) * E
        logging.debug("Constructed sparse Google matrix.")
    else:
        E = np.ones((n, n)) / n
        G = alpha * P + (1 - alpha) * E
        logging.debug("Constructed dense Google matrix.")
    return G

def power_method(G, tol=1e-8, max_iter=1000, verbose=False):
    """
    Applies the power method to find the dominant eigenvector of G.

    Parameters:
    - G (numpy.ndarray or scipy.sparse.csr_matrix): Google matrix.
    - tol (float): Tolerance for convergence.
    - max_iter (int): Maximum number of iterations.
    - verbose (bool): If True, prints convergence information.

    Returns:
    - numpy.ndarray: PageRank vector.
    - list: List of L1 norms at each iteration.
    """
    n = G.shape[0]
    x = np.ones(n) / n
    norms = []
    for iteration in range(1, max_iter + 1):
        x_next = G @ x
        norm = np.linalg.norm(x_next - x, 1)
        norms.append(norm)
        if verbose:
            logging.info(f"Iteration {iteration}: L1 norm = {norm:.6e}")
        if norm < tol:
            logging.info(f"Power method converged after {iteration} iterations.")
            break
        x = x_next
    else:
        logging.warning("Power method did not converge within the maximum number of iterations.")
    return x / x.sum(), norms

def plot_pagerank(x, title="PageRank Distribution", save_path=None):
    """
    Plots the PageRank vector.

    Parameters:
    - x (numpy.ndarray): PageRank vector.
    - title (str): Title of the plot.
    - save_path (str, optional): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(x)), x, color='skyblue')
    plt.xlabel('Website Index')
    plt.ylabel('PageRank')
    plt.title(title)
    plt.xticks(range(len(x)))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"PageRank distribution plot saved to {save_path}")
    if ENABLE_VISUALIZE:
        plt.show()
    plt.close()

def plot_convergence(norms, title="Power Method Convergence", save_path=None):
    """
    Plots the convergence of the power method.

    Parameters:
    - norms (list): List of L1 norms per iteration.
    - title (str): Title of the plot.
    - save_path (str, optional): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(norms, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('L1 Norm')
    plt.title(title)
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Convergence plot saved to {save_path}")
    if ENABLE_PLOT_CONVERGENCE and ENABLE_VISUALIZE:
        plt.show()
    plt.close()

def visualize_graph(A, pagerank, title="Web Graph with PageRank", save_path=None):
    """
    Visualizes the web graph with node sizes proportional to PageRank.

    Parameters:
    - A (numpy.ndarray or scipy.sparse.csr_matrix): Adjacency matrix.
    - pagerank (numpy.ndarray): PageRank vector.
    - title (str): Title of the graph.
    - save_path (str, optional): File path to save the graph.
    """
    if isinstance(A, csr_matrix):
        G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
    node_sizes = 1000 * pagerank  # Scale for visibility
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=pagerank, cmap=plt.cm.Blues)
    edges = nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowstyle='->', arrowsize=10)
    labels = nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Web graph visualization saved to {save_path}")
    if ENABLE_PLOT_GRAPH and ENABLE_VISUALIZE:
        plt.show()
    plt.close()

def print_matrix(name, matrix):
    """
    Prints a matrix with a given name.

    Parameters:
    - name (str): Name of the matrix.
    - matrix (numpy.ndarray or scipy.sparse.csr_matrix): Matrix to print.
    """
    print(f"\n{name}:")
    if isinstance(matrix, csr_matrix):
        print(matrix.toarray())
    else:
        print(matrix)

def main():
    # ========================
    # Set Up Logging
    # ========================
    # Already configured at the top based on ENABLE_VERBOSE

    # ========================
    # Calculate Probability p
    # ========================
    if NUM_WEBSITES <= 1:
        logging.error("Number of websites (n) must be greater than 1.")
        sys.exit(1)
    
    logging.info(f"Number of websites: {NUM_WEBSITES}")
    
    # ========================
    # Ensure Save Directory Exists
    # ========================
    if SAVE_DIRECTORY:
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        logging.info(f"Plots will be saved to directory: {SAVE_DIRECTORY}")

    # ========================
    # Generate Adjacency Matrix
    # ========================
    A = generate_preferential_adjacency(NUM_WEBSITES, beta=BETA, seed=RANDOM_SEED, sparse=USE_SPARSE_MATRICES)
    print_matrix("Adjacency Matrix (A)", A)

    # ========================
    # Convert to Row-Stochastic Matrix
    # ========================
    P = adjacency_to_stochastic(A)
    print_matrix("Row-Stochastic Matrix (P)", P)

    # ========================
    # Construct Google Matrix
    # ========================
    G = google_matrix(P, alpha=DAMPING_FACTOR, sparse=USE_SPARSE_MATRICES)
    print_matrix("Google Matrix (G)", G)

    # ========================
    # Compute PageRank using Power Method
    # ========================
    x_pagerank, norms = power_method(G.T, tol=TOLERANCE, max_iter=MAX_ITERATIONS, verbose=ENABLE_VERBOSE)

    # ========================
    # Output Results
    # ========================
    print("\nPageRank Vector:")
    for i, rank in enumerate(x_pagerank):
        print(f"Website {i+1}: {rank:.6f}")
    print(f"\nSum of PageRank entries = {x_pagerank.sum():.4f}")

    # ========================
    # Generate and Save Plots
    # ========================
    if ENABLE_VISUALIZE:
        if SAVE_DIRECTORY:
            pagerank_plot_path = os.path.join(SAVE_DIRECTORY, "pagerank_distribution.png")
        else:
            pagerank_plot_path = None
        plot_pagerank(x_pagerank, title="PageRank Distribution (Incremental Model)", save_path=pagerank_plot_path)

    if ENABLE_PLOT_CONVERGENCE:
        if SAVE_DIRECTORY:
            convergence_plot_path = os.path.join(SAVE_DIRECTORY, "convergence_plot.png")
        else:
            convergence_plot_path = None
        plot_convergence(norms, title="Power Method Convergence (Incremental Model)", save_path=convergence_plot_path)

    if ENABLE_PLOT_GRAPH:
        if SAVE_DIRECTORY:
            graph_plot_path = os.path.join(SAVE_DIRECTORY, "web_graph.png")
        else:
            graph_plot_path = None
        visualize_graph(A, x_pagerank, title="Web Graph with PageRank (Incremental Model)", save_path=graph_plot_path)

if __name__ == "__main__":
    main()