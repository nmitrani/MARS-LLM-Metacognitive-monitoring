"""
Setup script for the LLM Logprob Reporting Experiment
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def download_model():
    """Download the default model."""
    print("Downloading default model (this may take a few minutes)...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "microsoft/DialoGPT-small"
        print(f"Downloading {model_name}...")

        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer downloaded")

        # Download model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ Model downloaded")

        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False


def run_test():
    """Run a quick test to verify installation."""
    print("Running installation test...")
    try:
        subprocess.check_call([sys.executable, "test_experiment.py"])
        print("✓ Installation test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("LLM LOGPROB REPORTING EXPERIMENT - SETUP")
    print("=" * 60)

    steps = [
        ("Installing requirements", install_requirements),
        ("Downloading model", download_model),
        ("Running test", run_test),
    ]

    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"\nSetup failed at: {step_name}")
            print("Please check the error messages above and try again.")
            return False

    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYou can now run the experiment using:")
    print("  python run_experiment.py")
    print("\nOr run examples using:")
    print("  python example_usage.py")
    print("\nFor help, run:")
    print("  python run_experiment.py --help")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
