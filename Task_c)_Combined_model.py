import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# ========================
# Configuration Variables
# ========================
NUM_WEBSITES = 20         # Number of websites (n)
DAMPING_FACTOR = 0.2      # Damping factor (alpha)
TOLERANCE = 1e-8          # Tolerance for power method convergence
MAX_ITERATIONS = 50       # Maximum iterations for power method
RANDOM_SEED = 42          # Seed for reproducibility
ENABLE_VISUALIZE = True   # Enable visualization of results
SAVE_DIRECTORY = "plots_uniform_vs_incremental_models"  

# ========================
# Configure Logging
# ========================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ========================
# Helper Functions
# ========================
def create_uniform_links(nodes, link_prob, seed=None):
    """Generates an adjacency matrix for the uniform model."""
    if seed:
        np.random.seed(seed)
    adj_matrix = (np.random.rand(nodes, nodes) < link_prob).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def create_preferential_links(nodes, avg_links=3, seed=None):
    """Creates an adjacency matrix based on incremental linking probabilities."""
    if seed:
        np.random.seed(seed)
    adj_matrix = np.zeros((nodes, nodes), dtype=int)
    for current in range(1, nodes):
        prob = min(1.0, avg_links / current)
        links = np.random.rand(current) < prob
        adj_matrix[current, :current] = links.astype(int)
    return adj_matrix

def normalize_rows(matrix):
    """Converts an adjacency matrix into a row-stochastic matrix."""
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return matrix / row_sums[:, None]

def compute_google_matrix(stochastic_matrix, alpha):
    """Combines the stochastic matrix with teleportation probabilities."""
    size = stochastic_matrix.shape[0]
    teleportation = np.ones((size, size)) / size
    return alpha * stochastic_matrix + (1 - alpha) * teleportation

def calculate_pagerank(matrix, tolerance, iterations):
    """Applies the power method to compute the PageRank vector."""
    size = matrix.shape[0]
    pagerank_vector = np.ones(size) / size  # Uniform initial probability
    for step in range(iterations):
        updated_vector = matrix.T @ pagerank_vector
        if np.linalg.norm(updated_vector - pagerank_vector, 1) < tolerance:
            logging.info(f"Converged after {step + 1} iterations.")
            break
        pagerank_vector = updated_vector
    return pagerank_vector / pagerank_vector.sum()

def save_and_plot_pagerank(uniform_scores, incremental_scores, output_dir=None):
    """Saves and visualizes the comparison of PageRank scores."""
    plt.figure(figsize=(10, 6))
    plt.plot(uniform_scores, label="Uniform Model", marker='o', linestyle='-', color='blue')
    plt.plot(incremental_scores, label="Incremental Model", marker='x', linestyle='--', color='orange')
    plt.title("Comparison of PageRank Scores", fontsize=14)
    plt.xlabel("Node Index", fontsize=12)
    plt.ylabel("PageRank Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, "pagerank_comparison.png")
        plt.savefig(plot_file, dpi=300)
        logging.info(f"Plot saved at {plot_file}")

    if ENABLE_VISUALIZE:
        plt.show()

# ========================
# Main Execution
# ========================
def main():
    if SAVE_DIRECTORY:
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
        logging.info(f"Output directory set to: {SAVE_DIRECTORY}")

    # Generate adjacency matrices
    uniform_adj = create_uniform_links(NUM_WEBSITES, 3.0 / (NUM_WEBSITES - 1), seed=RANDOM_SEED)
    incremental_adj = create_preferential_links(NUM_WEBSITES, avg_links=3, seed=RANDOM_SEED)

    # Create Google matrices
    uniform_stochastic = normalize_rows(uniform_adj)
    incremental_stochastic = normalize_rows(incremental_adj)
    uniform_google = compute_google_matrix(uniform_stochastic, DAMPING_FACTOR)
    incremental_google = compute_google_matrix(incremental_stochastic, DAMPING_FACTOR)

    # Compute PageRank scores
    uniform_scores = calculate_pagerank(uniform_google, TOLERANCE, MAX_ITERATIONS)
    incremental_scores = calculate_pagerank(incremental_google, TOLERANCE, MAX_ITERATIONS)

    # Plot and save results
    save_and_plot_pagerank(uniform_scores, incremental_scores, output_dir=SAVE_DIRECTORY)

if __name__ == "__main__":
    main()