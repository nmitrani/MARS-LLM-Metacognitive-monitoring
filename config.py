"""
Configuration file for the LLM Logprob Reporting Experiment
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for the logprob reporting experiment."""

    # Model settings
    model_name: str = "microsoft/DialoGPT-small"  # Small model that fits in 8GB RAM
    device: str = "cpu"  # Force CPU usage for compatibility
    max_length: int = 512
    temperature: float = 0.0  # Deterministic sampling

    # Experiment settings
    num_icl_examples: int = 3
    num_test_examples: int = 10
    seed: int = 42

    # Dataset settings
    dataset_name: str = "mmlu"  # "mmlu" or "synthetic"
    dataset_subject: str = "abstract_algebra"  # For MMLU
    dataset_size: int = 50

    # Evaluation settings
    min_valid_responses: int = 5  # Minimum valid responses for meaningful evaluation

    # Output settings
    output_file: str = "logprob_experiment_results.json"
    verbose: bool = True


# Predefined configurations for different experiment scenarios
CONFIGS = {
    "quick": ExperimentConfig(num_icl_examples=2, num_test_examples=5, dataset_size=10),
    "standard": ExperimentConfig(
        num_icl_examples=3, num_test_examples=10, dataset_size=50
    ),
    "extensive": ExperimentConfig(
        num_icl_examples=5, num_test_examples=20, dataset_size=100
    ),
    "cpu_optimized": ExperimentConfig(
        model_name="microsoft/DialoGPT-small",
        device="cpu",
        max_length=256,
        num_icl_examples=2,
        num_test_examples=5,
    ),
    "synthetic_only": ExperimentConfig(
        dataset_name="synthetic", num_icl_examples=3, num_test_examples=10
    ),
}


def get_config(config_name: str = "standard") -> ExperimentConfig:
    """Get a predefined configuration by name."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config name: {config_name}. Available: {list(CONFIGS.keys())}"
        )

    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create a custom configuration with overridden values."""
    base_config = ExperimentConfig()

    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return base_config
