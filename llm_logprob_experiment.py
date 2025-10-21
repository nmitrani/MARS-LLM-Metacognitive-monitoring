"""
LLM Logprob Reporting Experiment

This experiment tests whether an LLM can accurately report its own logprobs
in a multi-turn conversation setting with in-context learning examples.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the logprob reporting experiment."""

    model_name: str = "microsoft/DialoGPT-small"  # Small model that fits in 8GB RAM
    max_length: int = 512
    num_icl_examples: int = 10
    num_test_examples: int = 10
    temperature: float = 0.0  # Deterministic sampling
    seed: int = 42
    device: str = "cpu"  # Force CPU usage for compatibility


class LLMWrapper:
    """Wrapper for LLM inference with logprob extraction capabilities."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU compatibility
            device_map="cpu",
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def get_logprobs(self, prompt: str, target_token: str) -> float:
        """Get log probability of a specific token given a prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the last token's logits
        last_token_logits = logits[0, -1, :]

        # Convert to probabilities
        probs = torch.softmax(last_token_logits, dim=-1)

        # Get the target token ID
        target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)[
            0
        ]

        # Return log probability
        return float(torch.log(probs[target_token_id]))

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response given a prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()


class DatasetLoader:
    """Load and prepare MCQ datasets for the experiment."""

    @staticmethod
    def load_mmlu_subset(subject: str = "abstract_algebra", num_examples: int = 50):
        """Load a subset of MMLU dataset for neutral MCQ testing."""
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            examples = []

            for i, example in enumerate(dataset):
                if i >= num_examples:
                    break

                question = example["question"]
                choices = example["choices"]  # Simplified - choices is already a list
                correct_answer = example["choices"][example["answer"]]

                # Format as MCQ with options and correct answer
                options_text = " ".join(
                    [f"({chr(65+j)}) {choice}" for j, choice in enumerate(choices)]
                )
                # Find the letter corresponding to the correct answer
                correct_letter = chr(65 + example["answer"])
                formatted_question = (
                    f"{question} {options_text} Answer: {correct_letter}"
                )
                examples.append(
                    {
                        "question": question,
                        "choices": choices,
                        "correct_answer": correct_answer,
                        "formatted": formatted_question,
                    }
                )

            return examples
        except Exception as e:
            logger.warning(f"Could not load MMLU dataset: {e}")
            return DatasetLoader._create_synthetic_dataset(num_examples)

    @staticmethod
    def _create_synthetic_dataset(num_examples: int = 50):
        """Create a synthetic MCQ dataset if MMLU is not available."""
        synthetic_examples = [
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "correct_answer": "4",
                "answer_index": 1,  # B is correct (index 1)
                "formatted": "What is 2 + 2? (A) 3 (B) 4 (C) 5 (D) 6 Answer: B",
            },
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "correct_answer": "Paris",
                "answer_index": 2,  # C is correct (index 2)
                "formatted": "What is the capital of France? (A) London (B) Berlin (C) Paris (D) Madrid Answer: C",
            },
            {
                "question": "What color is the sky?",
                "choices": ["Red", "Blue", "Green", "Yellow"],
                "correct_answer": "Blue",
                "answer_index": 1,  # B is correct (index 1)
                "formatted": "What color is the sky? (A) Red (B) Blue (C) Green (D) Yellow Answer: B",
            },
        ]

        # Repeat and vary the examples
        examples = []
        for i in range(num_examples):
            base_example = synthetic_examples[i % len(synthetic_examples)]
            examples.append(base_example.copy())

        return examples


class ConversationSimulator:
    """Simulate multi-turn conversations with ICL examples."""

    def __init__(self, llm_wrapper: LLMWrapper, config: ExperimentConfig):
        self.llm = llm_wrapper
        self.config = config

    def create_icl_prompt(self, icl_examples: List[Dict], test_question: str) -> str:
        """Create a prompt with in-context learning examples."""
        system_prompt = """You are participating in an experiment. In each turn, you will see examples of how to respond. Your task is to follow the same format when answering the final question.

Examples:"""

        prompt = system_prompt + "\n\n"

        # Add ICL examples
        for example in icl_examples:
            # Split from the end to get the last token as answer
            parts = example["formatted"].rsplit(" ", 1)
            question = parts[0]
            answer = parts[1]
            logprob = example["true_logprob"]
            prompt += f"User: Say something\n"
            prompt += f"Assistant: {question} {answer} {{{logprob:.4f}}}\n\n"

        # Add the test question
        prompt += f"User: Say something\n"
        prompt += f"Assistant: {test_question} "

        return prompt

    def run_experiment_turn(self, icl_examples: List[Dict], test_example: Dict) -> Dict:
        """Run a single turn of the experiment."""
        # Get true logprob for the test example
        # Split from the end to get the last token as answer
        parts = test_example["formatted"].rsplit(" ", 1)
        test_question = parts[0]
        test_answer = parts[1]

        # Create prompt to get true logprob
        true_logprob_prompt = f"{test_question} {test_answer}"
        true_logprob = self.llm.get_logprobs(true_logprob_prompt, test_answer)

        # Create ICL prompt
        icl_prompt = self.create_icl_prompt(icl_examples, test_question)

        # Generate response
        response = self.llm.generate_response(icl_prompt, max_length=50)

        # Parse reported logprob from response
        reported_logprob = self._extract_logprob_from_response(response)

        return {
            "test_question": test_question,
            "test_answer": test_answer,
            "true_logprob": true_logprob,
            "reported_logprob": reported_logprob,
            "response": response,
            "icl_prompt": icl_prompt,
        }

    def _extract_logprob_from_response(self, response: str) -> Optional[float]:
        """Extract logprob from the model's response."""
        try:
            # Look for pattern like {logprob} in the response
            import re

            pattern = r"\{([+-]?\d+\.?\d*)\}"
            matches = re.findall(pattern, response)

            if matches:
                return float(matches[-1])  # Take the last match
            else:
                return None
        except (ValueError, IndexError):
            return None


class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm = LLMWrapper(config.model_name, config.device)
        self.simulator = ConversationSimulator(self.llm, config)

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def run_experiment(self) -> Dict:
        """Run the complete experiment."""
        logger.info("Loading dataset...")
        dataset = DatasetLoader.load_mmlu_subset(
            num_examples=self.config.num_icl_examples + self.config.num_test_examples
        )

        # Split into ICL examples and test examples
        icl_examples = dataset[: self.config.num_icl_examples]
        test_examples = dataset[self.config.num_icl_examples :]

        # Get true logprobs for ICL examples
        logger.info("Computing true logprobs for ICL examples...")
        for example in icl_examples:
            # Split from the end to get the last token as answer
            parts = example["formatted"].rsplit(" ", 1)
            question = parts[0]
            answer = parts[1]
            logprob_prompt = f"{question} {answer}"
            example["true_logprob"] = self.llm.get_logprobs(logprob_prompt, answer)

        # Run experiment turns
        logger.info("Running experiment turns...")
        results = []

        for i, test_example in enumerate(test_examples):
            logger.info(f"Running turn {i+1}/{len(test_examples)}")
            result = self.simulator.run_experiment_turn(icl_examples, test_example)
            results.append(result)

        # Compute evaluation metrics
        metrics = self._compute_metrics(results)

        return {"config": self.config.__dict__, "results": results, "metrics": metrics}

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics."""
        valid_results = [r for r in results if r["reported_logprob"] is not None]

        if not valid_results:
            return {"error": "No valid reported logprobs found"}

        true_logprobs = [r["true_logprob"] for r in valid_results]
        reported_logprobs = [r["reported_logprob"] for r in valid_results]

        # Compute correlation
        correlation = np.corrcoef(true_logprobs, reported_logprobs)[0, 1]

        # Compute mean absolute error
        mae = np.mean(np.abs(np.array(true_logprobs) - np.array(reported_logprobs)))

        # Compute root mean square error
        rmse = np.sqrt(
            np.mean((np.array(true_logprobs) - np.array(reported_logprobs)) ** 2)
        )

        return {
            "num_valid_responses": len(valid_results),
            "num_total_responses": len(results),
            "correlation": float(correlation),
            "mae": float(mae),
            "rmse": float(rmse),
            "true_logprobs_mean": float(np.mean(true_logprobs)),
            "reported_logprobs_mean": float(np.mean(reported_logprobs)),
        }


def main():
    """Main function to run the experiment."""
    config = ExperimentConfig()

    logger.info("Starting LLM Logprob Reporting Experiment")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")

    runner = ExperimentRunner(config)
    results = runner.run_experiment()

    # Print results
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)

    metrics = results["metrics"]
    print(
        f"Valid responses: {metrics['num_valid_responses']}/{metrics['num_total_responses']}"
    )
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"True logprobs mean: {metrics['true_logprobs_mean']:.4f}")
    print(f"Reported logprobs mean: {metrics['reported_logprobs_mean']:.4f}")

    # Save detailed results
    output_file = "logprob_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
