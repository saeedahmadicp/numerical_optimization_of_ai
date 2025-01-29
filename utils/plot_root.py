# utils/plot_root.py

"""Plotting utilities for root-finding methods."""

from matplotlib import pyplot as plt

__all__ = ["plot_root"]


def plot_root(index, data):
    """Plot convergence of different root-finding methods.

    Args:
        index: Function index for plot title
        data: Dict of method results containing errors and iterations
    """
    # Add iteration indices to data
    for method_data in data.values():
        method_data["iterations"] = list(range(1, len(method_data["errors"]) + 1))

    # Create convergence plot
    plt.figure()
    for method, method_data in data.items():
        plt.plot(method_data["iterations"], method_data["errors"], label=method)

    plt.xlabel("Iterations")
    plt.ylabel("Absolute Errors")
    plt.title("Convergence Comparison")
    plt.legend(title="Root finding methods", loc="upper right")
    plt.show()
