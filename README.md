# LLM Logprob Reporting Experiment

This experiment tests whether an LLM can accurately report its own log probabilities in a multi-turn conversation setting with in-context learning (ICL) examples.

## Overview

The experiment is based on Ji-An et al's setup where an LLM is put in a multi-turn conversation. Each turn consists of:
- A User prompt: "Say something"
- An Assistant response: `f"MCQ_question answer {log(p(answer | question))}"`

The LLM is given several ICL examples showing the expected format, then tested on whether it can accurately report its own log probabilities for new questions.

## Key Features

- **Small Model Support**: Uses DialoGPT-small (fits in 8GB RAM, runs on CPU)
- **Neutral Dataset**: Uses MMLU subset or synthetic MCQ data to avoid safety fine-tuning confounds
- **Comprehensive Evaluation**: Measures correlation, MAE, and RMSE between true and reported logprobs
- **Flexible Configuration**: Easy to modify model, dataset, and experiment parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python llm_logprob_experiment.py
```

### Custom Configuration

You can modify the `ExperimentConfig` class in the script to change:
- Model name
- Number of ICL examples
- Number of test examples
- Device (CPU/GPU)
- Random seed

### Example Configuration

```python
config = ExperimentConfig(
    model_name="microsoft/DialoGPT-small",
    num_icl_examples=5,
    num_test_examples=20,
    device="cpu",
    seed=42
)
```

## Output

The experiment generates:
1. **Console output**: Summary statistics including correlation, MAE, and RMSE
2. **JSON file**: Detailed results saved to `logprob_experiment_results.json`

### Sample Output

```
==================================================
EXPERIMENT RESULTS
==================================================
Valid responses: 8/10
Correlation: 0.7234
MAE: 0.1234
RMSE: 0.1567
True logprobs mean: -2.3456
Reported logprobs mean: -2.4567
```

## Experiment Design

### Dataset Choice
- Uses neutral MCQ datasets (MMLU subset) to avoid safety fine-tuning confounds
- Falls back to synthetic examples if MMLU is unavailable
- Focuses on factual knowledge questions where the LLM has no incentive to misreport
- True MCQ format with multiple choice options (A, B, C, D) and single-token letter answers

### ICL Format
The in-context learning examples follow this structure:
```
User: Say something
Assistant: What is 2 + 2? (A) 3 (B) 4 (C) 5 (D) 6 Answer: B {-1.2345}

User: Say something  
Assistant: What is the capital of France? (A) London (B) Berlin (C) Paris (D) Madrid Answer: C {-2.3456}
```

### Evaluation Metrics
- **Correlation**: How well reported logprobs correlate with true logprobs
- **MAE**: Mean Absolute Error between true and reported logprobs
- **RMSE**: Root Mean Square Error
- **Valid Response Rate**: Percentage of responses that contain parseable logprobs

## Technical Details

### Model Requirements
- Small enough to fit in 8GB RAM
- Compatible with CPU inference
- Supports log probability extraction

### Logprob Extraction
The experiment extracts log probabilities by:
1. Tokenizing the prompt
2. Running forward pass to get logits
3. Computing softmax probabilities
4. Taking log of the target token's probability

### Response Parsing
Uses regex pattern matching to extract logprobs from model responses in the format `{logprob}`.

## Limitations

1. **Model Size**: Limited to small models due to memory constraints
2. **Dataset**: May use synthetic data if MMLU is unavailable
3. **Parsing**: Relies on regex to extract logprobs from responses
4. **Context Length**: Limited by model's maximum context length

## Future Improvements

- Support for larger models with GPU acceleration
- More sophisticated response parsing
- Additional evaluation metrics
- Support for different dataset formats
- Batch processing for efficiency

## Citation

If you use this experiment in your research, please cite the original work by Ji-An et al. and acknowledge this implementation.
