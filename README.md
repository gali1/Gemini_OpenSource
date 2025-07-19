# Gemini CLI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive AI assistant powered by the open-source Gemini PyTorch implementation. Gemini CLI provides a feature-rich command-line interface with multimodal capabilities, tool integration, and advanced conversation management.

![Gemini CLI Demo](gemini.png)

## ✨ Features

- **🤖 Interactive AI Chat**: Engage with the Gemini model through an intuitive CLI
- **🔧 Tool Integration**: Built-in tools for file operations, shell commands, web search, and more
- **💾 Persistent Memory**: Conversation history and context management across sessions
- **🎨 Customizable Themes**: Multiple color themes and UI customization options
- **🔒 Sandbox Execution**: Safe execution environment for code and shell commands
- **📁 File Discovery**: Intelligent file inclusion with pattern matching and gitignore support
- **🔄 Session Management**: Save, resume, and manage conversation sessions
- **⚡ Performance Tracking**: Built-in statistics and performance monitoring
- **🧩 Extension System**: Modular architecture with extension support
- **🌐 Multi-modal Support**: Text, image, audio, and video processing capabilities

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM (8GB+ recommended for larger models)
- **Storage**: 2GB+ free space for model weights and cache

### Optional Dependencies

- **Docker**: For sandbox execution (recommended)
- **Git**: For version control integration
- **CUDA**: For GPU acceleration (if available)

## 🚀 Installation

### Method 1: Using pip (Recommended)

```bash
# Install from PyPI (coming soon)
pip install gemini-cli

# Or install with all features
pip install gemini-cli[all]
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/kyegomez/Gemini.git
cd Gemini

# Create a virtual environment (recommended)
python -m venv gemini-env
source gemini-env/bin/activate  # On Windows: gemini-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Method 3: Development Installation

```bash
# Clone and install for development
git clone https://github.com/kyegomez/Gemini.git
cd Gemini

# Create virtual environment
python -m venv gemini-dev
source gemini-dev/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .[dev]
```

### Quick Setup Script

For a one-line installation:

```bash
curl -sSL https://raw.githubusercontent.com/kyegomez/Gemini/main/install.sh | bash
```

## ⚙️ Configuration

### First Run Setup

```bash
# Initialize configuration
gemini-cli --init

# Or start with interactive setup
gemini-cli
```

### Configuration File

Gemini CLI looks for configuration files in the following order:

1. `./gemini/settings.json` (project-specific)
2. `~/.gemini/settings.json` (user-specific)
3. `/etc/gemini-cli/settings.json` (system-wide)

### Basic Configuration

Create `~/.gemini/settings.json`:

```json
{
  "model": "gemini-torch",
  "maxTokens": 4096,
  "temperature": 0.7,
  "theme": "Default",
  "autoAccept": false,
  "sandbox": {
    "enabled": true,
    "type": "docker"
  },
  "memory": {
    "enabled": true,
    "maxMemoryFiles": 50
  }
}
```

### Environment Variables

```bash
# Model configuration
export GEMINI_MODEL="gemini-torch"
export GEMINI_MAX_TOKENS=4096
export GEMINI_TEMPERATURE=0.7

# Feature toggles
export GEMINI_AUTO_ACCEPT=false
export GEMINI_SANDBOX=true
export GEMINI_DEBUG=false

# Paths
export GEMINI_CONFIG_PATH="~/.gemini/settings.json"
```

## 🎯 Usage

### Basic Commands

```bash
# Start interactive mode
gemini-cli

# Single prompt mode
gemini-cli "Hello, how can you help me today?"

# Include files in context
gemini-cli "@README.md Explain this project"

# Execute with specific model
gemini-cli --model gemini-large "Analyze this code: @src/main.py"
```

### Interactive Mode Commands

Once in interactive mode, you can use these commands:

#### Slash Commands
- `/help` - Show available commands
- `/quit` - Exit the CLI
- `/clear` - Clear the screen
- `/stats` - Show session statistics
- `/theme [name]` - Change color theme
- `/tools` - List available tools
- `/memory show` - Display memory context
- `/chat save [name]` - Save current conversation
- `/restore [id]` - Restore from checkpoint

#### File Commands
- `@file.py` - Include file content in prompt
- `@src/*.py` - Include multiple files with glob patterns
- `@docs/` - Include all files in directory

#### Shell Commands
- `!ls -la` - Execute shell command
- `!` - Toggle shell mode

### Advanced Usage

#### Custom Configuration
```bash
# Use custom config file
gemini-cli --config my-config.json

# Override specific settings
gemini-cli --temperature 0.9 --max-tokens 8192

# Enable debug mode
gemini-cli --debug --verbose
```

#### Sandbox Mode
```bash
# Enable sandbox for safe execution
gemini-cli --sandbox

# Use specific sandbox image
gemini-cli --sandbox --config '{"sandbox": {"image": "custom-image"}}'
```

#### Memory Management
```bash
# Disable memory loading
gemini-cli --no-memory

# Add persistent memory
gemini-cli
> /memory add "I prefer Python over JavaScript"
```

## 🔧 Tools

Gemini CLI includes a rich set of built-in tools:

### File Operations
- `read_file` - Read file contents
- `write_file` - Create or modify files
- `list_directory` - Browse directories
- `glob` - Find files with patterns
- `search_file_content` - Search text in files

### Code Execution
- `run_shell_command` - Execute shell commands
- `run_python_code` - Execute Python code
- `run_javascript_code` - Execute JavaScript code

### Web Operations
- `web_fetch` - Fetch content from URLs
- `google_web_search` - Search the web

### Utility Tools
- `save_memory` - Save information to memory
- `compress_conversation` - Compress chat history

### Tool Usage Examples

```bash
# In interactive mode
> Can you read the contents of package.json?
# Gemini will use the read_file tool automatically

> Please create a Python script that prints "Hello World"
# Gemini will use write_file to create the script

> Search for TODO comments in my Python files
# Gemini will use search_file_content tool
```

## 🎨 Customization

### Themes

Available themes:
- `Default` - Standard color scheme
- `Dark` - Dark mode optimized
- `Light` - Light mode optimized
- `Monokai` - Developer-friendly
- `GitHub` - GitHub-inspired

```bash
# Change theme
gemini-cli --theme Dark

# List available themes
gemini-cli --list-themes
```

### Custom Themes

Create `~/.gemini/themes/mytheme.json`:

```json
{
  "name": "MyTheme",
  "description": "My custom theme",
  "colors": {
    "primary": "BLUE",
    "secondary": "CYAN",
    "success": "GREEN",
    "warning": "YELLOW",
    "error": "RED",
    "info": "BLUE",
    "prompt": "GREEN",
    "user_input": "WHITE",
    "ai_response": "CYAN",
    "tool_call": "MAGENTA",
    "system": "DIM",
    "accent": "BRIGHT_BLUE"
  }
}
```

### Extensions

Create custom extensions in `~/.gemini/extensions/`:

```python
# ~/.gemini/extensions/my-extension/extension.py
from gemini_cli.core.extensions import BaseExtension

class MyExtension(BaseExtension):
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.description = "My custom extension"

    async def initialize(self, cli_context):
        return True

    async def shutdown(self):
        return True

    def get_commands(self):
        return {
            "hello": self._hello_command
        }

    async def _hello_command(self, args):
        return f"Hello from {self.name}!"
```

## 📊 Monitoring & Statistics

### Session Statistics
```bash
# View current session stats
> /stats

# Export session data
gemini-cli --export-stats session.json
```

### Performance Monitoring
```bash
# Enable verbose logging
gemini-cli --verbose --log-level DEBUG

# Monitor memory usage
gemini-cli --debug
> /stats memory
```

### Health Checks
```bash
# Run system health check
gemini-cli --health-check

# Test tool functionality
gemini-cli --test-tools
```

## 🔒 Security & Sandboxing

### Sandbox Configuration

```json
{
  "sandbox": {
    "enabled": true,
    "type": "docker",
    "image": "gemini-cli-sandbox",
    "memoryLimit": "512m",
    "cpuLimit": "0.5",
    "allowNetwork": false,
    "readOnlyFilesystem": true
  }
}
```

### Safe Execution

```bash
# Enable automatic sandboxing
gemini-cli --sandbox

# Review commands before execution
gemini-cli --no-auto-accept
```

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Numpy installation fails with setup.py error
```bash
# Solution: Update pip and use wheel installations
pip install --upgrade pip setuptools wheel
pip install numpy --no-use-pep517
```

**Issue**: PyTorch CPU installation problems
```bash
# Solution: Install PyTorch separately first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gemini-cli
```

#### Runtime Issues

**Issue**: ModuleNotFoundError for gemini_torch
```bash
# Solution: Install in development mode
pip install -e .
```

**Issue**: Permission denied for sandbox
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

#### Configuration Issues

**Issue**: Config file not found
```bash
# Solution: Initialize configuration
gemini-cli --init
# Or create manually
mkdir -p ~/.gemini
echo '{}' > ~/.gemini/settings.json
```

### Debug Mode

```bash
# Enable detailed logging
gemini-cli --debug --verbose

# Check configuration
gemini-cli --config-info

# Validate installation
gemini-cli --health-check
```

### Getting Help

1. **Documentation**: Check the [GitHub Wiki](https://github.com/kyegomez/Gemini/wiki)
2. **Issues**: Report bugs on [GitHub Issues](https://github.com/kyegomez/Gemini/issues)
3. **Discussions**: Join [GitHub Discussions](https://github.com/kyegomez/Gemini/discussions)
4. **Discord**: Join our [Discord community](https://discord.gg/qUtxnK2NMf)

## 🧪 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/kyegomez/Gemini.git
cd Gemini

# Create development environment
python -m venv gemini-dev
source gemini-dev/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gemini_cli

# Run specific test file
pytest tests/test_cli.py

# Run in parallel
pytest -n auto
```

### Code Style

```bash
# Format code
black gemini_cli/
isort gemini_cli/

# Lint code
flake8 gemini_cli/
pylint gemini_cli/

# Type checking
mypy gemini_cli/
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e .[docs]

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000 -d _build/html
```

## 📝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Format code: `black . && isort .`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the [Gemini PyTorch](https://github.com/kyegomez/Gemini) implementation
- Inspired by the original [Gemini](https://deepmind.google/technologies/gemini/) by Google DeepMind
- Uses [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- Powered by [PyTorch](https://pytorch.org/) for model inference

## 📈 Roadmap

- [ ] Plugin marketplace
- [ ] Voice input/output support
- [ ] Enhanced multimodal capabilities
- [ ] Cloud deployment options
- [ ] Mobile companion app
- [ ] Advanced debugging tools
- [ ] Performance optimizations
- [ ] Multi-language support

---

**⭐ Star this repository if you find it helpful!**

For more information, visit our [GitHub repository](https://github.com/kyegomez/Gemini) or check out the [documentation](https://github.com/kyegomez/Gemini/wiki).