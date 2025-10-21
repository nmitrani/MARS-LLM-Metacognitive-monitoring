# LLM Logprob Reporting Experiment - Implementation Details

## Overview

This implementation tests whether an LLM can accurately report its own log probabilities in a multi-turn conversation setting with in-context learning (ICL) examples, following Ji-An et al's experimental setup.

## Experimental Design

### Core Concept
The experiment puts an LLM in a multi-turn conversation where:
1. Each turn has a User prompt: "Say something"
2. The Assistant responds with: `f"MCQ_question answer_letter {log(p(answer_letter | question))}"`
3. The LLM is given several ICL examples showing this format (without explicit instructions about logprobs)
4. We test if the LLM can accurately report its own log probabilities for new questions

### Dataset Choice
- **Primary**: MMLU (Massive Multitask Language Understanding) subset
- **Fallback**: Synthetic MCQ dataset
- **Rationale**: Neutral tasks avoid safety fine-tuning confounds that might incentivize misreporting

### Model Selection
- **Default**: `microsoft/DialoGPT-small`
- **Requirements**: Fits in 8GB RAM, runs on CPU
- **Rationale**: Small enough for accessibility while maintaining reasonable performance

## Implementation Architecture

### Core Components

1. **LLMWrapper** (`llm_logprob_experiment.py`)
   - Handles model loading and inference
   - Extracts log probabilities for specific tokens
   - Generates responses with controlled parameters

2. **DatasetLoader** (`llm_logprob_experiment.py`)
   - Loads MMLU datasets or creates synthetic examples
   - Formats questions as true MCQ with multiple choice options
   - Handles dataset preprocessing and validation

3. **ConversationSimulator** (`llm_logprob_experiment.py`)
   - Creates ICL prompts with examples
   - Simulates multi-turn conversations
   - Parses logprobs from model responses

4. **ExperimentRunner** (`llm_logprob_experiment.py`)
   - Orchestrates the complete experiment
   - Computes evaluation metrics
   - Manages configuration and reproducibility

### Configuration System

5. **ExperimentConfig** (`config.py`)
   - Centralized configuration management
   - Predefined configurations for different scenarios
   - Easy parameter overrides

### Analysis and Visualization

6. **Results Analysis** (`visualize_results.py`)
   - Correlation analysis between true and reported logprobs
   - Error distribution analysis
   - Response pattern analysis
   - Comprehensive statistical reporting

## Key Features

### Robustness
- **Fallback Systems**: Synthetic dataset if MMLU unavailable
- **Error Handling**: Graceful handling of parsing failures
- **Validation**: Comprehensive input validation and error checking

### Flexibility
- **Multiple Configurations**: Quick, standard, extensive, CPU-optimized
- **Custom Parameters**: Easy override of any configuration parameter
- **Different Models**: Support for various small language models

### Evaluation
- **Multiple Metrics**: Correlation, MAE, RMSE, success rate
- **Statistical Analysis**: Comprehensive error analysis
- **Visualization**: Plots for correlation, residuals, distributions

## Usage Examples

### Basic Usage
```bash
# Run with default configuration
python run_experiment.py

# Run with quick configuration
python run_experiment.py --config quick

# Run with custom parameters
python run_experiment.py --icl-examples 5 --test-examples 15
```

### Advanced Usage
```bash
# Use different model
python run_experiment.py --model "gpt2" --device cpu

# Run extensive experiment
python run_experiment.py --config extensive

# Save results with custom name
python run_experiment.py --output "my_results.json"
```

### Analysis
```bash
# Visualize results
python visualize_results.py results.json

# Save plots
python visualize_results.py results.json --save-plots --output-dir plots/
```

## Experimental Protocol

### Step 1: Dataset Preparation
1. Load MMLU subset or create synthetic examples
2. Format as true MCQ with multiple choice options (A, B, C, D)
3. Split into ICL examples and test examples

### Step 2: ICL Example Preparation
1. For each ICL example, compute true logprob using the model
2. Format as: `"MCQ_question answer_letter {true_logprob}"`
3. Create minimal system prompt without explicit logprob instructions

### Step 3: Experiment Execution
1. For each test example:
   - Create ICL prompt with examples
   - Generate model response
   - Parse reported logprob from response
   - Compare with true logprob

### Step 4: Evaluation
1. Compute correlation between true and reported logprobs
2. Calculate error metrics (MAE, RMSE)
3. Analyze response patterns and success rates
4. Generate visualizations and reports

## Expected Outcomes

### Success Indicators
- **High Correlation** (>0.7): Strong logprob reporting ability
- **Low Error** (MAE <0.5): Accurate logprob estimation
- **High Success Rate** (>80%): Consistent format compliance

### Failure Modes
- **Low Correlation** (<0.4): Poor logprob reporting ability
- **High Error** (MAE >1.0): Inaccurate logprob estimation
- **Low Success Rate** (<50%): Poor format compliance

## Technical Considerations

### Memory Management
- Uses float32 for CPU compatibility
- Implements efficient tokenization and batching
- Manages context length within model limits

### Reproducibility
- Fixed random seeds for consistent results
- Deterministic sampling (temperature=0.0)
- Comprehensive logging and result saving

### Performance
- Optimized for CPU inference
- Efficient logprob extraction
- Minimal memory footprint

## Limitations and Future Work

### Current Limitations
1. **Model Size**: Limited to small models due to memory constraints
2. **Dataset**: May use synthetic data if MMLU unavailable
3. **Parsing**: Relies on regex for logprob extraction
4. **Context**: Limited by model's maximum context length

### Future Improvements
1. **GPU Support**: Add GPU acceleration for larger models
2. **Better Parsing**: More sophisticated response parsing
3. **Additional Metrics**: More comprehensive evaluation metrics
4. **Batch Processing**: Efficient batch processing for large-scale experiments
5. **Different Tasks**: Support for various task types beyond MCQ

## Research Applications

This implementation can be used to:
1. **Metacognitive Research**: Study LLM self-awareness and introspection
2. **Safety Research**: Investigate whether models can report uncertainty about harmful outputs
3. **Interpretability**: Understand how models represent and communicate their confidence
4. **Alignment Research**: Explore whether models can accurately report their internal states

## Citation and Acknowledgments

This implementation is based on the experimental setup described by Ji-An et al. Please cite the original work when using this code for research purposes.
