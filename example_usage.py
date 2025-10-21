"""
Example usage of the LLM Logprob Reporting Experiment
"""

import logging
from llm_logprob_experiment import ExperimentRunner, DatasetLoader
from config import get_config, create_custom_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_quick_experiment():
    """Run a quick experiment with minimal examples."""
    logger.info("Running Quick Experiment Example")

    # Use the predefined "quick" configuration
    config = get_config("quick")

    # Run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    # Print key results
    metrics = results["metrics"]
    print(f"Quick experiment results:")
    print(
        f"  Valid responses: {metrics['num_valid_responses']}/{metrics['num_total_responses']}"
    )
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")

    return results


def example_custom_experiment():
    """Run an experiment with custom configuration."""
    logger.info("Running Custom Experiment Example")

    # Create a custom configuration
    config = create_custom_config(
        model_name="microsoft/DialoGPT-small",
        device="cpu",
        num_icl_examples=2,
        num_test_examples=5,
        seed=123,
    )

    # Run the experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    # Print key results
    metrics = results["metrics"]
    print(f"Custom experiment results:")
    print(
        f"  Valid responses: {metrics['num_valid_responses']}/{metrics['num_total_responses']}"
    )
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")

    return results


def example_synthetic_dataset():
    """Demonstrate using synthetic dataset."""
    logger.info("Running Synthetic Dataset Example")

    # Load synthetic dataset
    dataset = DatasetLoader._create_synthetic_dataset(num_examples=10)

    print("Synthetic dataset examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"  {i+1}. {example['formatted']}")

    # Run experiment with synthetic data
    config = get_config("synthetic_only")
    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    return results


def example_analyze_results():
    """Demonstrate how to analyze experiment results."""
    logger.info("Analyzing Results Example")

    # Run a quick experiment
    config = get_config("quick")
    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    # Analyze individual results
    print("\nIndividual test results:")
    for i, result in enumerate(results["results"][:3]):  # Show first 3
        print(f"\nTest {i+1}:")
        print(f"  Question: {result['test_question']}")
        print(f"  Answer: {result['test_answer']}")
        print(f"  True logprob: {result['true_logprob']:.4f}")
        print(f"  Reported logprob: {result['reported_logprob']}")
        print(f"  Response: {result['response'][:100]}...")

    # Analyze metrics
    metrics = results["metrics"]
    print(f"\nOverall metrics:")
    print(
        f"  Success rate: {metrics['num_valid_responses']/metrics['num_total_responses']*100:.1f}%"
    )
    print(f"  Correlation: {metrics['correlation']:.4f}")

    if metrics["correlation"] > 0.7:
        print("  Interpretation: Good logprob reporting ability")
    elif metrics["correlation"] > 0.4:
        print("  Interpretation: Moderate logprob reporting ability")
    else:
        print("  Interpretation: Poor logprob reporting ability")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM LOGPROB REPORTING EXPERIMENT - EXAMPLES")
    print("=" * 60)

    examples = [
        ("Quick Experiment", example_quick_experiment),
        ("Custom Configuration", example_custom_experiment),
        ("Synthetic Dataset", example_synthetic_dataset),
        ("Results Analysis", example_analyze_results),
    ]

    for name, example_func in examples:
        print(f"\n{'-'*40}")
        print(f"Running: {name}")
        print(f"{'-'*40}")

        try:
            example_func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")

    print(f"\n{'='*60}")
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
