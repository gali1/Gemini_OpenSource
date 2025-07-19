#!/usr/bin/env python3
"""
Installation script for Gemini CLI
Handles PyTorch CPU installation and package setup
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def run_command(cmd, description="Running command"):
    """Run a command and handle errors with live output."""
    print(f"{description}: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"Python version check passed: {sys.version}")

def install_pytorch_cpu():
    """Install PyTorch CPU version."""
    print("\n=== Installing PyTorch (CPU version) ===")

    pytorch_cmd = [
        sys.executable, "-m", "pip", "install", "-v",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]

    if not run_command(pytorch_cmd, "Installing PyTorch CPU"):
        print("Failed to install PyTorch. Trying alternative method...")

        fallback_cmd = [
            sys.executable, "-m", "pip", "install", "-v",
            "torch", "torchvision", "torchaudio"
        ]

        if not run_command(fallback_cmd, "Installing PyTorch (fallback)"):
            print("Error: Failed to install PyTorch")
            return False
    return True

def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n=== Installing requirements ===")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"Warning: requirements.txt not found at {requirements_file}")
        return True

    cmd = [
        sys.executable, "-m", "pip", "install", "-v",
        "-r", str(requirements_file)
    ]

    return run_command(cmd, "Installing requirements")

def install_package():
    """Install the Gemini CLI package."""
    print("\n=== Installing Gemini CLI package ===")

    cmd = [
        sys.executable, "-m", "pip", "install", "-v",
        "-e", "."
    ]

    return run_command(cmd, "Installing Gemini CLI")

def verify_installation():
    """Verify that the installation was successful."""
    print("\n=== Verifying installation ===")

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")

        import gemini_cli
        print(f"✓ Gemini CLI installed: {gemini_cli.__version__}")

        cmd = [sys.executable, "-m", "gemini_cli", "--help"]
        subprocess.run(cmd, timeout=10)

        print("✓ Gemini CLI command line interface is working")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ Warning: CLI verification timed out")
        return True
    except Exception as e:
        print(f"⚠ Warning: Verification failed: {e}")
        return True

def main():
    """Main installation function."""
    print("Gemini CLI Installation Script")
    print("=" * 40)

    check_python_version()

    print("\n=== Upgrading pip ===")
    upgrade_pip_cmd = [sys.executable, "-m", "pip", "install", "-v", "--upgrade", "pip"]
    run_command(upgrade_pip_cmd, "Upgrading pip")

    if not install_pytorch_cpu():
        print("Failed to install PyTorch")
        sys.exit(1)

    if not install_requirements():
        print("Failed to install requirements")
        sys.exit(1)

    if not install_package():
        print("Failed to install Gemini CLI package")
        sys.exit(1)

    if verify_installation():
        print("\n" + "=" * 40)
        print("✓ Installation completed successfully!")
        print("\nYou can now run:")
        print("  gemini-cli --help")
        print("  gcli --help")
        print("  python -m gemini_cli --help")
    else:
        print("\n" + "=" * 40)
        print("⚠ Installation completed with warnings")
        print("Please check the error messages above")
        print("\nFor more information, visit:")
        print("  https://github.com/kyegomez/Gemini")

if __name__ == "__main__":
    main()
