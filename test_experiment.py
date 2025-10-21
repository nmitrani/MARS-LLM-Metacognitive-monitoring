"""
Test script for the LLM Logprob Reporting Experiment
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_logprob_experiment import (
    ExperimentConfig,
    LLMWrapper,
    DatasetLoader,
    ConversationSimulator,
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_llm_wrapper():
    """Test the LLM wrapper functionality."""
    logger.info("Testing LLM Wrapper...")

    try:
        config = ExperimentConfig()
        llm = LLMWrapper(config.model_name, config.device)

        # Test logprob extraction
        test_prompt = "What is 2 + 2? (A) 3 (B) 4 (C) 5 (D) 6 Answer: B"
        test_token = "B"
        logprob = llm.get_logprobs(test_prompt, test_token)

        logger.info(f"Logprob for '{test_token}' given '{test_prompt}': {logprob}")

        # Test response generation
        response = llm.generate_response("Hello, how are you?", max_length=20)
        logger.info(f"Generated response: {response}")

        return True

    except Exception as e:
        logger.error(f"LLM Wrapper test failed: {e}")
        return False


def test_dataset_loader():
    """Test the dataset loader functionality."""
    logger.info("Testing Dataset Loader...")

    try:
        # Test synthetic dataset creation
        dataset = DatasetLoader._create_synthetic_dataset(num_examples=5)

        logger.info(f"Loaded {len(dataset)} synthetic examples")
        for i, example in enumerate(dataset[:2]):
            logger.info(f"Example {i+1}: {example['formatted']}")

        return True

    except Exception as e:
        logger.error(f"Dataset Loader test failed: {e}")
        return False


def test_conversation_simulator():
    """Test the conversation simulator functionality."""
    logger.info("Testing Conversation Simulator...")

    try:
        config = ExperimentConfig()
        llm = LLMWrapper(config.model_name, config.device)
        simulator = ConversationSimulator(llm, config)

        # Create test ICL examples
        icl_examples = [
            {
                "formatted": "What is 2 + 2? (A) 3 (B) 4 (C) 5 (D) 6 Answer: B",
                "true_logprob": -1.5,
            },
            {
                "formatted": "What is 3 + 3? (A) 5 (B) 6 (C) 7 (D) 8 Answer: B",
                "true_logprob": -2.0,
            },
        ]

        # Create test example
        test_example = {"formatted": "What is 4 + 4? (A) 6 (B) 7 (C) 8 (D) 9 Answer: C"}

        # Test ICL prompt creation
        icl_prompt = simulator.create_icl_prompt(icl_examples, "What is 4 + 4?")
        logger.info(f"Generated ICL prompt length: {len(icl_prompt)}")

        return True

    except Exception as e:
        logger.error(f"Conversation Simulator test failed: {e}")
        return False


def run_quick_experiment():
    """Run a quick version of the experiment with minimal examples."""
    logger.info("Running Quick Experiment...")

    try:
        config = ExperimentConfig()
        config.num_icl_examples = 2
        config.num_test_examples = 3

        from llm_logprob_experiment import ExperimentRunner

        runner = ExperimentRunner(config)

        # Load minimal dataset
        dataset = DatasetLoader._create_synthetic_dataset(num_examples=5)

        # Split into ICL and test examples
        icl_examples = dataset[: config.num_icl_examples]
        test_examples = dataset[config.num_icl_examples :]

        # Get true logprobs for ICL examples
        for example in icl_examples:
            # Split from the end to get the last token as answer
            parts = example["formatted"].rsplit(" ", 1)
            question = parts[0]
            answer = parts[1]
            logprob_prompt = f"{question} {answer}"
            example["true_logprob"] = runner.llm.get_logprobs(logprob_prompt, answer)

        # Run one test turn
        result = runner.simulator.run_experiment_turn(icl_examples, test_examples[0])

        logger.info(f"Test result: {result}")

        return True

    except Exception as e:
        logger.error(f"Quick experiment failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting LLM Logprob Experiment Tests")

    tests = [
        ("LLM Wrapper", test_llm_wrapper),
        ("Dataset Loader", test_dataset_loader),
        ("Conversation Simulator", test_conversation_simulator),
        ("Quick Experiment", run_quick_experiment),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")

        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"{test_name} Test: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} Test FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("All tests passed! The experiment is ready to run.")
    else:
        logger.warning("Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
