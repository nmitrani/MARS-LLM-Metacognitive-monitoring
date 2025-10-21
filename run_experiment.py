"""
Main script to run the LLM Logprob Reporting Experiment
"""

import argparse
import json
import logging
from typing import Dict, Any

from llm_logprob_experiment import ExperimentRunner, DatasetLoader
from config import get_config, create_custom_config, ExperimentConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the experiment with the given configuration."""
    logger.info("Starting LLM Logprob Reporting Experiment")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"ICL Examples: {config.num_icl_examples}")
    logger.info(f"Test Examples: {config.num_test_examples}")

    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    return results


def print_results(results: Dict[str, Any]):
    """Print experiment results in a formatted way."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    metrics = results["metrics"]

    if "error" in metrics:
        print(f"ERROR: {metrics['error']}")
        return

    print(
        f"Valid responses: {metrics['num_valid_responses']}/{metrics['num_total_responses']}"
    )
    print(
        f"Success rate: {metrics['num_valid_responses']/metrics['num_total_responses']*100:.1f}%"
    )
    print()
    print("Correlation Analysis:")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print()
    print("Logprob Statistics:")
    print(f"  True logprobs mean: {metrics['true_logprobs_mean']:.4f}")
    print(f"  Reported logprobs mean: {metrics['reported_logprobs_mean']:.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    correlation = metrics["correlation"]
    if correlation > 0.8:
        interpretation = (
            "EXCELLENT: Strong correlation indicates good logprob reporting ability"
        )
    elif correlation > 0.6:
        interpretation = (
            "GOOD: Moderate correlation suggests some logprob reporting ability"
        )
    elif correlation > 0.4:
        interpretation = (
            "FAIR: Weak correlation indicates limited logprob reporting ability"
        )
    else:
        interpretation = (
            "POOR: Very weak correlation suggests poor logprob reporting ability"
        )

    print(f"Correlation interpretation: {interpretation}")

    success_rate = metrics["num_valid_responses"] / metrics["num_total_responses"]
    if success_rate > 0.8:
        format_interpretation = (
            "EXCELLENT: Model consistently follows the required format"
        )
    elif success_rate > 0.6:
        format_interpretation = "GOOD: Model mostly follows the required format"
    elif success_rate > 0.4:
        format_interpretation = "FAIR: Model sometimes follows the required format"
    else:
        format_interpretation = "POOR: Model rarely follows the required format"

    print(f"Format compliance: {format_interpretation}")


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to a JSON file."""
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run LLM Logprob Reporting Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="standard",
        choices=["quick", "standard", "extensive", "cpu_optimized", "synthetic_only"],
        help="Predefined configuration to use",
    )

    parser.add_argument("--model", type=str, help="Override model name")

    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], help="Override device"
    )

    parser.add_argument(
        "--icl-examples", type=int, help="Override number of ICL examples"
    )

    parser.add_argument(
        "--test-examples", type=int, help="Override number of test examples"
    )

    parser.add_argument("--output", type=str, help="Override output file path")

    parser.add_argument("--seed", type=int, help="Override random seed")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get base configuration
    config = get_config(args.config)

    # Apply overrides
    overrides = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.icl_examples:
        overrides["num_icl_examples"] = args.icl_examples
    if args.test_examples:
        overrides["num_test_examples"] = args.test_examples
    if args.output:
        overrides["output_file"] = args.output
    if args.seed:
        overrides["seed"] = args.seed

    if overrides:
        config = create_custom_config(**overrides)

    try:
        # Run experiment
        results = run_experiment(config)

        # Print results
        print_results(results)

        # Save results
        save_results(results, config.output_file)

        logger.info("Experiment completed successfully!")

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
