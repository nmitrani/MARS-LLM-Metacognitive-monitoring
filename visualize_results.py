"""
Visualization script for LLM Logprob Reporting Experiment results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, List
import argparse


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def plot_correlation(results: Dict[str, Any], save_path: str = None):
    """Plot correlation between true and reported logprobs."""
    valid_results = [r for r in results["results"] if r["reported_logprob"] is not None]

    if not valid_results:
        print("No valid results to plot")
        return

    true_logprobs = [r["true_logprob"] for r in valid_results]
    reported_logprobs = [r["reported_logprob"] for r in valid_results]

    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(true_logprobs, reported_logprobs, alpha=0.7)
    plt.plot(
        [min(true_logprobs), max(true_logprobs)],
        [min(true_logprobs), max(true_logprobs)],
        "r--",
        alpha=0.8,
    )
    plt.xlabel("True Logprobs")
    plt.ylabel("Reported Logprobs")
    plt.title("True vs Reported Logprobs")

    # Add correlation coefficient
    correlation = np.corrcoef(true_logprobs, reported_logprobs)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = np.array(reported_logprobs) - np.array(true_logprobs)
    plt.scatter(true_logprobs, residuals, alpha=0.7)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("True Logprobs")
    plt.ylabel("Residuals (Reported - True)")
    plt.title("Residuals Plot")

    # Distribution of true logprobs
    plt.subplot(2, 2, 3)
    plt.hist(true_logprobs, bins=15, alpha=0.7, label="True", density=True)
    plt.hist(reported_logprobs, bins=15, alpha=0.7, label="Reported", density=True)
    plt.xlabel("Logprob Value")
    plt.ylabel("Density")
    plt.title("Distribution of Logprobs")
    plt.legend()

    # Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=15, alpha=0.7)
    plt.xlabel("Error (Reported - True)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_response_analysis(results: Dict[str, Any], save_path: str = None):
    """Plot analysis of response patterns."""
    valid_results = [r for r in results["results"] if r["reported_logprob"] is not None]
    invalid_results = [r for r in results["results"] if r["reported_logprob"] is None]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Success rate
    success_rate = len(valid_results) / len(results["results"])
    axes[0, 0].pie(
        [success_rate, 1 - success_rate],
        labels=["Valid Responses", "Invalid Responses"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0, 0].set_title("Response Success Rate")

    # Response length distribution
    response_lengths = [len(r["response"]) for r in results["results"]]
    axes[0, 1].hist(response_lengths, bins=15, alpha=0.7)
    axes[0, 1].set_xlabel("Response Length (characters)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Response Length Distribution")

    # Error magnitude vs response length
    if valid_results:
        errors = [abs(r["reported_logprob"] - r["true_logprob"]) for r in valid_results]
        lengths = [len(r["response"]) for r in valid_results]
        axes[1, 0].scatter(lengths, errors, alpha=0.7)
        axes[1, 0].set_xlabel("Response Length")
        axes[1, 0].set_ylabel("Absolute Error")
        axes[1, 0].set_title("Error vs Response Length")

    # Sample responses
    axes[1, 1].axis("off")
    sample_text = "Sample Responses:\n\n"
    for i, result in enumerate(results["results"][:3]):
        sample_text += f"Test {i+1}:\n"
        sample_text += f"Q: {result['test_question'][:50]}...\n"
        sample_text += f"R: {result['response'][:100]}...\n\n"

    axes[1, 1].text(
        0.05,
        0.95,
        sample_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Response analysis plot saved to: {save_path}")
    else:
        plt.show()


def print_detailed_analysis(results: Dict[str, Any]):
    """Print detailed analysis of results."""
    print("\n" + "=" * 60)
    print("DETAILED RESULTS ANALYSIS")
    print("=" * 60)

    metrics = results["metrics"]
    valid_results = [r for r in results["results"] if r["reported_logprob"] is not None]

    print(f"Total experiments: {len(results['results'])}")
    print(f"Valid responses: {len(valid_results)}")
    print(f"Success rate: {len(valid_results)/len(results['results'])*100:.1f}%")

    if valid_results:
        print(f"\nCorrelation Analysis:")
        print(f"  Pearson correlation: {metrics['correlation']:.4f}")
        print(f"  Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"  Root Mean Square Error: {metrics['rmse']:.4f}")

        # Additional statistics
        true_logprobs = [r["true_logprob"] for r in valid_results]
        reported_logprobs = [r["reported_logprob"] for r in valid_results]
        errors = [abs(r["reported_logprob"] - r["true_logprob"]) for r in valid_results]

        print(f"\nError Statistics:")
        print(f"  Mean error: {np.mean(errors):.4f}")
        print(f"  Median error: {np.median(errors):.4f}")
        print(f"  Std error: {np.std(errors):.4f}")
        print(f"  Max error: {np.max(errors):.4f}")

        print(f"\nLogprob Statistics:")
        print(
            f"  True logprobs - Mean: {np.mean(true_logprobs):.4f}, Std: {np.std(true_logprobs):.4f}"
        )
        print(
            f"  Reported logprobs - Mean: {np.mean(reported_logprobs):.4f}, Std: {np.std(reported_logprobs):.4f}"
        )

    # Analyze invalid responses
    invalid_results = [r for r in results["results"] if r["reported_logprob"] is None]
    if invalid_results:
        print(f"\nInvalid Response Analysis:")
        print(f"  Number of invalid responses: {len(invalid_results)}")
        print(f"  Common patterns in invalid responses:")

        # Look for common patterns in invalid responses
        response_patterns = {}
        for result in invalid_results:
            response = result["response"]
            if "{" in response and "}" in response:
                response_patterns["Contains braces"] = (
                    response_patterns.get("Contains braces", 0) + 1
                )
            if len(response) < 10:
                response_patterns["Very short"] = (
                    response_patterns.get("Very short", 0) + 1
                )

        for pattern, count in response_patterns.items():
            print(f"    {pattern}: {count}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results_file", help="Path to results JSON file")
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots instead of showing"
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save plots")

    args = parser.parse_args()

    # Load results
    try:
        results = load_results(args.results_file)
    except FileNotFoundError:
        print(f"Error: Could not find results file: {args.results_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {args.results_file}")
        return

    # Print detailed analysis
    print_detailed_analysis(results)

    # Create plots
    if args.save_plots:
        plot_correlation(results, f"{args.output_dir}/correlation_analysis.png")
        plot_response_analysis(results, f"{args.output_dir}/response_analysis.png")
    else:
        plot_correlation(results)
        plot_response_analysis(results)


if __name__ == "__main__":
    main()
