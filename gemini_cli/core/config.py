"""
Configuration management for Gemini CLI
Handles settings.json files, environment variables, and configuration overrides
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import copy


class ConfigManager:
    """Manages CLI configuration from multiple sources with proper precedence."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self.overrides = {}
        self.env_vars = {}

        # Load configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from all sources with proper precedence."""
        # 1. Default values
        self.config = self._get_default_config()

        # 2. System settings file
        self._load_system_config()

        # 3. User settings file
        self._load_user_config()

        # 4. Project settings file
        self._load_project_config()

        # 5. Custom config file if specified
        if self.config_path:
            self._load_custom_config()

        # 6. Environment variables
        self._load_env_config()

        # 7. Apply any overrides
        if self.overrides:
            self._deep_merge(self.config, self.overrides)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "model": "gemini-torch",
            "maxTokens": 32768,
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,

            # UI settings
            "theme": "Default",
            "hideTips": False,
            "hideBanner": False,

            # Tool settings
            "autoAccept": False,
            "coreTools": [],  # Empty means all tools available
            "excludeTools": [],

            # File handling
            "fileFiltering": {
                "respectGitIgnore": True,
                "enableRecursiveFileSearch": True
            },

            # Context settings
            "contextFileName": "GEMINI.md",
            "maxSessionTurns": -1,  # Unlimited

            # Safety and sandboxing
            "sandbox": {
                "enabled": False,
                "image": "gemini-cli-sandbox",
                "type": "docker"
            },

            # Memory settings
            "memory": {
                "enabled": True,
                "maxMemoryFiles": 50,
                "memoryFileName": "GEMINI.md"
            },

            # Checkpointing
            "checkpointing": {
                "enabled": False,
                "maxCheckpoints": 10
            },

            # Telemetry
            "telemetry": {
                "enabled": False,
                "target": "local",
                "otlpEndpoint": "http://localhost:4317",
                "logPrompts": True
            },

            # Usage statistics
            "usageStatisticsEnabled": True,

            # MCP servers
            "mcpServers": {},
            "allowMCPServers": [],
            "excludeMCPServers": [],

            # Extensions
            "extensions": {
                "enabled": [],
                "disabled": []
            },

            # Tool summarization
            "summarizeToolOutput": {},

            # Editor preferences
            "preferredEditor": "vscode",

            # Bug reporting
            "bugCommand": {
                "urlTemplate": "https://github.com/kyegomez/Gemini/issues/new?title={title}&info={info}"
            },

            # Authentication (for future API integrations)
            "auth": {
                "type": "local",  # local, api_key, oauth
                "apiKey": None,
                "endpoint": None
            },

            # Development and debugging
            "debug": False,
            "verbose": False,
            "logLevel": "INFO"
        }

    def _load_system_config(self):
        """Load system-wide configuration."""
        system_paths = [
            Path("/etc/gemini-cli/settings.json"),  # Linux
            Path("C:/ProgramData/gemini-cli/settings.json"),  # Windows
            Path("/Library/Application Support/GeminiCli/settings.json")  # macOS
        ]

        for path in system_paths:
            if path.exists():
                self._load_config_file(path)
                break

    def _load_user_config(self):
        """Load user-specific configuration."""
        user_config_path = Path.home() / ".gemini" / "settings.json"
        if user_config_path.exists():
            self._load_config_file(user_config_path)

    def _load_project_config(self):
        """Load project-specific configuration."""
        # Look for .gemini/settings.json in current directory and parents
        current_path = Path.cwd()

        while current_path != current_path.parent:
            config_path = current_path / ".gemini" / "settings.json"
            if config_path.exists():
                self._load_config_file(config_path)
                break

            # Stop at git root or home directory
            if (current_path / ".git").exists() or current_path == Path.home():
                break

            current_path = current_path.parent

    def _load_custom_config(self):
        """Load custom configuration file."""
        config_path = Path(self.config_path)
        if config_path.exists():
            self._load_config_file(config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def _load_config_file(self, path: Path):
        """Load configuration from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Resolve environment variables in string values
            config_data = self._resolve_env_vars(config_data)

            # Merge with existing config
            self._deep_merge(self.config, config_data)

        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file {path}: {e}")
        except Exception as e:
            print(f"Warning: Could not load config file {path}: {e}")

    def _resolve_env_vars(self, data: Any) -> Any:
        """Recursively resolve environment variables in configuration values."""
        if isinstance(data, dict):
            return {key: self._resolve_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_env_vars(data)
        else:
            return data

    def _substitute_env_vars(self, text: str) -> str:
        """Substitute environment variables in text using $VAR or ${VAR} syntax."""
        def replacer(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))

        # Match $VAR_NAME or ${VAR_NAME}
        pattern = r'\$(?:([A-Za-z_][A-Za-z0-9_]*)|{([A-Za-z_][A-Za-z0-9_]*)})'
        return re.sub(pattern, replacer, text)

    def _load_env_config(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "GEMINI_MODEL": "model",
            "GEMINI_API_KEY": "auth.apiKey",
            "GEMINI_ENDPOINT": "auth.endpoint",
            "GEMINI_DEBUG": "debug",
            "GEMINI_SANDBOX": "sandbox.enabled",
            "GEMINI_SANDBOX_IMAGE": "sandbox.image",
            "GEMINI_THEME": "theme",
            "GEMINI_AUTO_ACCEPT": "autoAccept",
            "GEMINI_MAX_TOKENS": "maxTokens",
            "GEMINI_TEMPERATURE": "temperature",
            "GEMINI_TELEMETRY": "telemetry.enabled",
            "GEMINI_TELEMETRY_TARGET": "telemetry.target",
            "GEMINI_TELEMETRY_ENDPOINT": "telemetry.otlpEndpoint",
            "GEMINI_LOG_LEVEL": "logLevel",
            "OTLP_EXPORTER_OTLP_ENDPOINT": "telemetry.otlpEndpoint",
            "NO_COLOR": "_noColor",  # Special handling
            "CLI_TITLE": "_cliTitle",  # Special handling
            "EDITOR": "preferredEditor"
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string values to appropriate types
                value = self._convert_env_value(value, config_path)
                self._set_nested_value(self.config, config_path, value)

    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if config_path in ["debug", "autoAccept", "sandbox.enabled", "telemetry.enabled",
                          "fileFiltering.respectGitIgnore", "fileFiltering.enableRecursiveFileSearch",
                          "hideTips", "hideBanner", "usageStatisticsEnabled"]:
            return value.lower() in ("true", "1", "yes", "on")

        # Integer values
        if config_path in ["maxTokens", "maxSessionTurns", "temperature", "topP", "topK"]:
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value

        # Special sandbox handling
        if config_path == "sandbox.enabled":
            if value.lower() in ("true", "1", "yes", "docker", "podman"):
                return True
            return False

        return value

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set a nested value in a dictionary using dot notation."""
        keys = path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return copy.deepcopy(self.config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            keys = key.split('.')
            current = self.config

            for k in keys:
                current = current[k]

            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation."""
        self._set_nested_value(self.config, key, value)

    def apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides (typically from command line)."""
        self.overrides = copy.deepcopy(overrides)
        self._deep_merge(self.config, self.overrides)

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration and persist to user config file."""
        self._deep_merge(self.config, updates)
        self._save_user_config(updates)

    def _save_user_config(self, updates: Dict[str, Any]):
        """Save updates to user configuration file."""
        try:
            config_dir = Path.home() / ".gemini"
            config_dir.mkdir(exist_ok=True)

            config_file = config_dir / "settings.json"

            # Load existing config or create new
            existing_config = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)

            # Merge updates
            self._deep_merge(existing_config, updates)

            # Save back to file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Could not save user configuration: {e}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check required fields
        if not self.get("model"):
            issues.append("Model not specified")

        # Validate numeric ranges
        max_tokens = self.get("maxTokens")
        if max_tokens and (max_tokens < 1 or max_tokens > 1000000):
            issues.append("maxTokens must be between 1 and 1000000")

        temperature = self.get("temperature")
        if temperature and (temperature < 0.0 or temperature > 2.0):
            issues.append("temperature must be between 0.0 and 2.0")

        # Validate tool configuration
        core_tools = self.get("coreTools", [])
        exclude_tools = self.get("excludeTools", [])

        if core_tools and exclude_tools:
            overlapping = set(core_tools) & set(exclude_tools)
            if overlapping:
                issues.append(f"Tools cannot be both included and excluded: {overlapping}")

        # Validate MCP server configuration
        mcp_servers = self.get("mcpServers", {})
        for server_name, config in mcp_servers.items():
            if not isinstance(config, dict):
                issues.append(f"MCP server '{server_name}' configuration must be an object")
                continue

            # Check that server has a command or URL
            if not any(key in config for key in ["command", "url", "httpUrl"]):
                issues.append(f"MCP server '{server_name}' must specify command, url, or httpUrl")

        # Validate sandbox configuration
        sandbox_config = self.get("sandbox", {})
        if sandbox_config.get("enabled"):
            sandbox_type = sandbox_config.get("type", "docker")
            if sandbox_type not in ["docker", "podman", "sandbox-exec"]:
                issues.append(f"Invalid sandbox type: {sandbox_type}")

        # Validate telemetry configuration
        telemetry_config = self.get("telemetry", {})
        if telemetry_config.get("enabled"):
            target = telemetry_config.get("target")
            if target not in ["local", "gcp"]:
                issues.append(f"Invalid telemetry target: {target}")

        return issues

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool-specific configuration."""
        return {
            "coreTools": self.get("coreTools", []),
            "excludeTools": self.get("excludeTools", []),
            "autoAccept": self.get("autoAccept", False),
            "sandbox": self.get("sandbox", {}),
            "summarizeToolOutput": self.get("summarizeToolOutput", {})
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            "model": self.get("model"),
            "maxTokens": self.get("maxTokens"),
            "temperature": self.get("temperature"),
            "topP": self.get("topP"),
            "topK": self.get("topK"),
            "auth": self.get("auth", {})
        }

    def get_file_discovery_config(self) -> Dict[str, Any]:
        """Get file discovery configuration."""
        return {
            "respectGitIgnore": self.get("fileFiltering.respectGitIgnore", True),
            "enableRecursiveFileSearch": self.get("fileFiltering.enableRecursiveFileSearch", True),
            "contextFileName": self.get("contextFileName", "GEMINI.md")
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory management configuration."""
        return {
            "enabled": self.get("memory.enabled", True),
            "maxMemoryFiles": self.get("memory.maxMemoryFiles", 50),
            "memoryFileName": self.get("memory.memoryFileName", "GEMINI.md"),
            "contextFileName": self.get("contextFileName", "GEMINI.md")
        }

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "sandbox": "sandbox.enabled",
            "telemetry": "telemetry.enabled",
            "checkpointing": "checkpointing.enabled",
            "memory": "memory.enabled",
            "usageStats": "usageStatisticsEnabled",
            "debug": "debug",
            "autoAccept": "autoAccept"
        }

        if feature in feature_map:
            return self.get(feature_map[feature], False)

        return False

    def get_env_file_paths(self) -> List[Path]:
        """Get potential .env file paths in search order."""
        paths = []

        # Start from current directory and go up
        current = Path.cwd()
        while current != current.parent:
            # Check for .gemini/.env first, then .env
            gemini_env = current / ".gemini" / ".env"
            if gemini_env.exists():
                paths.append(gemini_env)
                break

            regular_env = current / ".env"
            if regular_env.exists():
                paths.append(regular_env)
                break

            # Stop at git root
            if (current / ".git").exists():
                break

            current = current.parent

        # Fall back to home directory
        if not paths:
            home_gemini_env = Path.home() / ".gemini" / ".env"
            home_env = Path.home() / ".env"

            if home_gemini_env.exists():
                paths.append(home_gemini_env)
            elif home_env.exists():
                paths.append(home_env)

        return paths

    def load_env_file(self):
        """Load environment variables from .env file."""
        env_paths = self.get_env_file_paths()

        for env_path in env_paths:
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')

                            # Only set if not already in environment
                            if key not in os.environ:
                                os.environ[key] = value

                break  # Use first found file

            except Exception as e:
                print(f"Warning: Could not load .env file {env_path}: {e}")

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(model={self.get('model')}, theme={self.get('theme')})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ConfigManager(config_keys={list(self.config.keys())})"