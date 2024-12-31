import numpy as np
import matplotlib.pyplot as plt

def build_stochastic_matrix(links, n):
    """
    Build a column-stochastic matrix A from a link dictionary.
    Each column i represents the probability distribution for page i's out-links.

    Parameters:
    -----------
    links : dict
        Dictionary where links[i] is a list of nodes that page i points to.
    n : int
        Number of total pages (nodes).

    Returns:
    --------
    A : (n x n) numpy array
        The column-stochastic matrix (each column sums to 1 if not dangling).
    """
    A = np.zeros((n, n))
    for i in range(n):
        outgoing = links.get(i, [])
        d_i = len(outgoing)
        if d_i > 0:
            for j in outgoing:
                A[j, i] = 1.0 / d_i
    return A

def steady_state(A, tol=1e-9, max_iter=1000):
    """
    Compute the steady-state vector of a column-stochastic matrix A
    by power iteration.

    Parameters:
    -----------
    A : (n x n) numpy array
        Column-stochastic matrix.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    x : (n,) numpy array
        Steady-state vector.
    """
    n = A.shape[0]
    # Start with a uniform distribution
    x = np.ones(n) / n

    for iteration in range(max_iter):
        x_new = A @ x
        # Check for convergence using L1 norm
        if np.linalg.norm(x_new - x, 1) < tol:
            print(f"Converged after {iteration+1} iterations.")
            return x_new
        x = x_new

    print("Reached maximum iterations without full convergence.")
    return x

def power_method(A, x0=None, max_iter=100, tol=1e-12):
    """
    Basic power method to approximate x satisfying A x = x.

    Parameters:
    -----------
    A : (n x n) numpy array
        Square matrix.
    x0 : (n,) numpy array, optional
        Initial vector. If None, starts with uniform distribution.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.

    Returns:
    --------
    x : (n,) numpy array
        Dominant eigenvector.
    """
    n = A.shape[0]
    if x0 is None:
        x = np.ones(n) / n
    else:
        x = x0 / np.sum(x0)
    for iteration in range(max_iter):
        x_new = A @ x
        if np.linalg.norm(x_new - x, 1) < tol:
            print(f"Power method converged after {iteration+1} iterations.")
            return x_new / np.sum(x_new)
        x = x_new
    print("Power method reached maximum iterations without full convergence.")
    return x / np.sum(x)

def pagerank(A, p=0.15, max_iter=1000, tol=1e-9):
    """
    Compute the PageRank vector using the damping factor p.
    Handles dangling nodes by assigning uniform probabilities.

    Parameters:
    -----------
    A : (n x n) numpy array
        Column-stochastic matrix representing link structure.
    p : float
        Damping factor (probability of teleportation).
    max_iter : int
        Maximum number of iterations for the power method.
    tol : float
        Tolerance for convergence.

    Returns:
    --------
    x : (n,) numpy array
        PageRank vector.
    """
    n = A.shape[0]
    # Handle dangling nodes: replace zero columns with 1/n
    for i in range(n):
        if np.allclose(A[:, i], 0):
            A[:, i] = 1.0 / n

    # Construct the Google matrix M
    M = (1 - p) * A + (p / n) * np.ones((n, n))

    return steady_state(M, tol=tol, max_iter=max_iter)

def simulate_random_surfer(A, steps=20, start=0):
    """
    Simulate a random surfer (no teleportation) for a given number of steps.

    Parameters:
    -----------
    A : (n x n) numpy array
        Column-stochastic matrix representing link structure.
    steps : int
        Number of steps to simulate.
    start : int
        Starting page index.

    Returns:
    --------
    path : list
        Sequence of visited pages.
    """
    n = A.shape[0]
    path = [start]
    current = start
    for step in range(steps):
        probs = A[:, current]
        # If dangling, choose randomly among all pages
        if np.allclose(probs, 0):
            current = np.random.randint(n)
        else:
            current = np.random.choice(range(n), p=probs)
        path.append(current)
    return path

def plot_convergence(distributions, n):
    """
    Plot the convergence of PageRank distribution over iterations.

    Parameters:
    -----------
    distributions : (k x n) numpy array
        Array where each row represents the PageRank vector at an iteration.
    n : int
        Number of pages.
    """
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(distributions[:, i], label=f"Page {i+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("Convergence of PageRank Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate:
      1. Computing steady-state distributions without damping.
      2. Computing PageRank with damping.
      3. Visualizing convergence of PageRank.
      4. Simulating a random surfer.
    """
    # Define a sample web graph with 5 pages (0-based indexing)
    links = {
        0: [1],       # Page 0 -> Page 1
        1: [2, 4],    # Page 1 -> Pages 2, 4
        2: [0],       # Page 2 -> Page 0
        3: [1],       # Page 3 -> Page 1
        4: [3, 1]     # Page 4 -> Pages 3, 1
    }
    n = 5  # Number of pages

    # Build the stochastic matrix A
    A = build_stochastic_matrix(links, n)
    print("Stochastic Matrix A (columns sum to 1 if not dangling):\n", A)

    # 1. Compute steady-state without damping
    print("\n--- Steady-State Distribution (No Damping) ---")
    x_ss = steady_state(A)
    print("Steady state (no damping):", x_ss)

    # 2. Compute PageRank with damping
    print("\n--- PageRank with Damping ---")
    p = 0.15
    x_pr = pagerank(A.copy(), p=p)
    print(f"PageRank (with damping p={p}):", x_pr)

    # 3. Visualize convergence of PageRank
    print("\n--- Plotting Convergence of PageRank ---")
    # Reconstruct M for iterative convergence plotting
    for i in range(n):
        if np.allclose(A[:, i], 0):
            A[:, i] = 1.0 / n  # Handle dangling nodes

    M = (1 - p) * A + (p / n) * np.ones((n, n))
    x = np.ones(n) / n  # Initialize with uniform distribution

    distributions = [x.copy()]
    for iteration in range(30):
        x = M @ x
        x /= np.sum(x)  # Normalize to ensure probabilities sum to 1
        distributions.append(x.copy())

    distributions = np.array(distributions)
    plot_convergence(distributions, n)

    # 4. Simulate a random surfer
    print("\n--- Simulating a Random Surfer (No Teleportation) ---")
    steps = 20
    start_page = 0
    path = simulate_random_surfer(A, steps=steps, start=start_page)
    print(f"Random Surfer Path (starting at Page {start_page}):", path)

if __name__ == "__main__":
    main()