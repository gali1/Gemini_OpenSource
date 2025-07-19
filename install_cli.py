#!/usr/bin/env python3
"""
Complete installation and setup script for Gemini CLI
Handles all dependencies, configuration, and system setup
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiCLIInstaller:
    """Complete installer for Gemini CLI with all features."""

    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.install_dir = Path.cwd()
        self.home_dir = Path.home()
        self.config_dir = self.home_dir / ".gemini"

    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        logger.info("Checking system requirements...")

        # Check Python version
        if self.python_version < (3, 8):
            logger.error("Python 3.8 or higher required")
            return False

        # Check for pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"],
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("pip is not available")
            return False

        logger.info("✓ System requirements met")
        return True

    def setup_environment(self):
        """Setup the environment and directories."""
        logger.info("Setting up environment...")

        # Create config directory
        self.config_dir.mkdir(exist_ok=True)

        # Create subdirectories
        subdirs = [
            "sessions", "conversations", "checkpoints", "extensions",
            "themes", "tmp", "telemetry", "cache"
        ]

        for subdir in subdirs:
            (self.config_dir / subdir).mkdir(exist_ok=True)

        logger.info("✓ Environment setup complete")

    def install_dependencies(self) -> bool:
        """Install all required dependencies."""
        logger.info("Installing dependencies...")

        # Core dependencies with proper version handling
        dependencies = [
            "aiohttp>=3.8.0,<4.0.0",
            "aiofiles>=22.1.0,<24.0.0",
            "async-timeout>=4.0.0,<5.0.0",
            "torch>=2.0.0,<3.0.0",
            "torchvision>=0.15.0,<1.0.0",
            "torchaudio>=2.0.0,<3.0.0",
            "einops>=0.6.0,<1.0.0",
            "transformers>=4.30.0,<5.0.0",
            "sentencepiece>=0.1.99,<1.0.0",
            "tokenizers>=0.13.0,<1.0.0",
            "rich>=13.0.0,<14.0.0",
            "click>=8.0.0,<9.0.0",
            "prompt-toolkit>=3.0.0,<4.0.0",
            "numpy>=1.21.0,<2.0.0",
            "pandas>=1.5.0,<3.0.0",
            "pillow>=9.0.0,<11.0.0",
            "requests>=2.28.0,<3.0.0",
            "chardet>=5.0.0,<6.0.0",
            "psutil>=5.9.0,<6.0.0",
            "pyyaml>=6.0.0,<7.0.0",
        ]

        # Try to install zeta-torch and ring-attention-pytorch
        optional_deps = [
            "zeta-torch",
            "ring-attention-pytorch",
        ]

        try:
            # Install core dependencies
            for dep in dependencies:
                logger.info(f"Installing {dep}")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)

            # Try optional dependencies
            for dep in optional_deps:
                try:
                    logger.info(f"Installing optional dependency {dep}")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", dep
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    logger.warning(f"Optional dependency {dep} failed to install, continuing...")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

        logger.info("✓ Dependencies installed successfully")
        return True

    def create_default_config(self):
        """Create default configuration files."""
        logger.info("Creating default configuration...")

        default_config = {
            "model": "gemini-torch",
            "maxTokens": 4096,
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
            "theme": "Default",
            "autoAccept": False,
            "debug": False,
            "verbose": False,
            "logLevel": "INFO",
            "memory": {
                "enabled": True,
                "maxMemoryFiles": 50,
                "memoryFileName": "GEMINI.md"
            },
            "checkpointing": {
                "enabled": False,
                "maxCheckpoints": 10,
                "backend": "git"
            },
            "telemetry": {
                "enabled": False,
                "target": "local"
            },
            "sandbox": {
                "enabled": False,
                "type": "docker",
                "image": "gemini-cli-sandbox"
            },
            "fileFiltering": {
                "respectGitIgnore": True,
                "enableRecursiveFileSearch": True
            },
            "extensions": {
                "enabled": [],
                "disabled": []
            },
            "mcpServers": {},
            "coreTools": [],
            "excludeTools": []
        }

        config_file = self.config_dir / "settings.json"
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        logger.info("✓ Default configuration created")

    def setup_shell_integration(self):
        """Setup shell integration and aliases."""
        logger.info("Setting up shell integration...")

        # Create shell scripts
        shell_script_content = """#!/bin/bash
# Gemini CLI Shell Integration

# Aliases
alias gcli="python -m gemini_cli"
alias gemini="python -m gemini_cli"
alias gemini-cli="python -m gemini_cli"

# Functions
gemini_help() {
    python -m gemini_cli --help
}

gemini_tools() {
    python -m gemini_cli --list-tools
}

gemini_health() {
    python -m gemini_cli --health-check
}

export GEMINI_CLI_INSTALLED=1
"""

        # Save shell integration script
        shell_script = self.config_dir / "shell_integration.sh"
        with open(shell_script, 'w') as f:
            f.write(shell_script_content)

        # Make executable
        shell_script.chmod(0o755)

        logger.info("✓ Shell integration setup complete")

    def create_entry_points(self):
        """Create entry point scripts."""
        logger.info("Creating entry points...")

        # Create a launcher script
        launcher_content = f"""#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the Gemini CLI to Python path
gemini_path = Path(__file__).parent
sys.path.insert(0, str(gemini_path))

try:
    from gemini_cli.main import main
    sys.exit(main())
except ImportError as e:
    print(f"Error importing Gemini CLI: {{e}}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"Error running Gemini CLI: {{e}}")
    sys.exit(1)
"""

        # Create launcher in install directory
        launcher_path = self.install_dir / "gcli"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        launcher_path.chmod(0o755)

        # Also create in user bin if possible
        user_bin = self.home_dir / ".local" / "bin"
        if user_bin.exists() or self._try_create_dir(user_bin):
            user_launcher = user_bin / "gcli"
            shutil.copy2(launcher_path, user_launcher)
            user_launcher.chmod(0o755)

        logger.info("✓ Entry points created")

    def _try_create_dir(self, path: Path) -> bool:
        """Try to create directory."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def setup_development_mode(self):
        """Setup development mode installation."""
        logger.info("Setting up development mode...")

        try:
            # Install in development mode
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], check=True, cwd=self.install_dir)

            logger.info("✓ Development mode setup complete")

        except subprocess.CalledProcessError:
            logger.warning("Development mode setup failed, using manual setup")

    def verify_installation(self) -> bool:
        """Verify the installation works."""
        logger.info("Verifying installation...")

        try:
            # Test basic import
            import gemini_cli
            from gemini_cli.main import main
            from gemini_cli.core.cli import GeminiCLI
            from gemini_cli.core.config import ConfigManager

            # Test configuration loading
            config_manager = ConfigManager()
            config = config_manager.get_config()

            logger.info("✓ Installation verified successfully")
            return True

        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False

    def create_example_files(self):
        """Create example files and documentation."""
        logger.info("Creating example files...")

        # Create example GEMINI.md
        example_gemini_md = """# Gemini CLI Context

This is your personal context file for Gemini CLI. You can add information here that you want the AI to remember across sessions.

## About Me
- Name: [Your Name]
- Role: [Your Role]
- Preferences: [Your Preferences]

## Current Project
- Project: [Current Project Name]
- Description: [Brief Description]
- Tech Stack: [Technologies Used]

## Added Memories
<!-- Gemini CLI will add memories here automatically -->

"""

        example_file = self.config_dir / "GEMINI.md"
        if not example_file.exists():
            with open(example_file, 'w') as f:
                f.write(example_gemini_md)

        # Create usage examples
        examples_dir = self.config_dir / "examples"
        examples_dir.mkdir(exist_ok=True)

        usage_examples = """# Gemini CLI Usage Examples

## Basic Usage
```bash
# Start interactive mode
gcli

# Single prompt
gcli "Explain how Python decorators work"

# Include files in context
gcli "@src/main.py Explain this code"

# Health check
gcli --health-check
```

## Advanced Features
```bash
# Enable sandbox mode
gcli --sandbox

# Use specific model
gcli --model gemini-large

# Enable debug mode
gcli --debug

# List available tools
gcli --list-tools
```

## In-CLI Commands
```
/help                 - Show help
/tools               - List tools
/memory add "fact"   - Add to memory
/stats               - Show statistics
/theme Dark          - Change theme
@file.py            - Include file
!command            - Run shell command
```
"""

        with open(examples_dir / "usage.md", 'w') as f:
            f.write(usage_examples)

        logger.info("✓ Example files created")

    def print_installation_summary(self):
        """Print installation summary and next steps."""
        logger.info("\n" + "="*60)
        logger.info("🎉 Gemini CLI Installation Complete!")
        logger.info("="*60)

        print(f"""
Configuration Directory: {self.config_dir}
Launcher Script: {self.install_dir}/gcli

Quick Start:
1. Run: ./gcli --health-check
2. Start interactive mode: ./gcli
3. Get help: ./gcli --help

Available Commands:
- gcli                    # Interactive mode
- gcli "your prompt"      # Single prompt
- gcli --health-check     # System check
- gcli --list-tools       # Show tools

Shell Integration:
Add to your shell profile (~/.bashrc or ~/.zshrc):
source {self.config_dir}/shell_integration.sh

Configuration:
Edit {self.config_dir}/settings.json to customize settings.

Documentation:
See {self.config_dir}/examples/ for usage examples.
""")

    def run_installation(self) -> bool:
        """Run the complete installation process."""
        logger.info("Starting Gemini CLI installation...")

        try:
            # Check system requirements
            if not self.check_system_requirements():
                return False

            # Setup environment
            self.setup_environment()

            # Install dependencies
            if not self.install_dependencies():
                return False

            # Create configuration
            self.create_default_config()

            # Setup shell integration
            self.setup_shell_integration()

            # Create entry points
            self.create_entry_points()

            # Try development mode setup
            self.setup_development_mode()

            # Create examples
            self.create_example_files()

            # Verify installation
            if not self.verify_installation():
                logger.warning("Installation verification failed, but basic setup is complete")

            # Print summary
            self.print_installation_summary()

            return True

        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False


def main():
    """Main installation function."""
    installer = GeminiCLIInstaller()

    if installer.run_installation():
        print("\n✅ Installation completed successfully!")
        print("Run './gcli --help' to get started.")
        sys.exit(0)
    else:
        print("\n❌ Installation failed!")
        print("Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()