"""
Extension management for Gemini CLI
Handles loading, managing, and executing extensions
"""

import json
import asyncio
import importlib.util
import sys
from typing import Dict, Any, List, Optional, Set, Callable, Union
from pathlib import Path
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .exceptions import GeminiCLIError, ExtensionError


@dataclass
class ExtensionInfo:
    """Information about an extension."""
    name: str
    version: str
    description: str
    enabled: bool
    path: Path
    config: Dict[str, Any]
    dependencies: List[str]
    mcp_servers: Dict[str, Any]
    context_files: List[Path]
    exclude_tools: List[str]


class BaseExtension(ABC):
    """Base class for Gemini CLI extensions."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = False
        self.logger = logging.getLogger(f"extension.{name}")

    @abstractmethod
    async def initialize(self, cli_context: Dict[str, Any]) -> bool:
        """Initialize the extension."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the extension."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get extension information."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "description": getattr(self, "description", ""),
            "author": getattr(self, "author", ""),
            "homepage": getattr(self, "homepage", "")
        }

    def get_commands(self) -> Dict[str, Callable]:
        """Get extension commands."""
        return {}

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get extension tools."""
        return []

    def get_hooks(self) -> Dict[str, List[Callable]]:
        """Get extension hooks."""
        return {}


class ExtensionManager:
    """Manages CLI extensions and their lifecycle."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extension storage
        self.extensions: Dict[str, ExtensionInfo] = {}
        self.loaded_extensions: Dict[str, BaseExtension] = {}
        self.extension_commands: Dict[str, Callable] = {}
        self.extension_hooks: Dict[str, List[Callable]] = {}

        # Configuration
        self.enabled_extensions = set(config.get("extensions", {}).get("enabled", []))
        self.disabled_extensions = set(config.get("extensions", {}).get("disabled", []))

        # Extension directories
        self.user_extensions_dir = Path.home() / ".gemini" / "extensions"
        self.workspace_extensions_dir = Path.cwd() / ".gemini" / "extensions"

        # Ensure directories exist
        self.user_extensions_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_extensions_dir.mkdir(parents=True, exist_ok=True)

    async def load_extensions(self):
        """Load all available extensions."""
        self.logger.info("Loading extensions...")

        try:
            # Discover extensions
            await self._discover_extensions()

            # Load enabled extensions
            await self._load_enabled_extensions()

            self.logger.info(f"Loaded {len(self.loaded_extensions)} extensions")

        except Exception as e:
            self.logger.error(f"Failed to load extensions: {e}")
            raise GeminiCLIError(f"Extension loading failed: {e}")

    async def _discover_extensions(self):
        """Discover available extensions."""
        # Search in workspace directory first (higher priority)
        await self._discover_extensions_in_directory(self.workspace_extensions_dir, "workspace")

        # Search in user directory
        await self._discover_extensions_in_directory(self.user_extensions_dir, "user")

    async def _discover_extensions_in_directory(self, directory: Path, source: str):
        """Discover extensions in a specific directory."""
        if not directory.exists():
            return

        try:
            for extension_dir in directory.iterdir():
                if not extension_dir.is_dir():
                    continue

                # Look for gemini-extension.json
                config_file = extension_dir / "gemini-extension.json"
                if not config_file.exists():
                    continue

                try:
                    extension_info = await self._load_extension_config(config_file, extension_dir, source)
                    if extension_info:
                        # Workspace extensions override user extensions
                        if extension_info.name not in self.extensions or source == "workspace":
                            self.extensions[extension_info.name] = extension_info
                            self.logger.debug(f"Discovered extension: {extension_info.name} from {source}")

                except Exception as e:
                    self.logger.warning(f"Failed to load extension config from {config_file}: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Failed to scan extension directory {directory}: {e}")

    async def _load_extension_config(self, config_file: Path, extension_dir: Path, source: str) -> Optional[ExtensionInfo]:
        """Load extension configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            name = config.get("name")
            if not name:
                self.logger.warning(f"Extension config missing name: {config_file}")
                return None

            version = config.get("version", "1.0.0")
            description = config.get("description", "")

            # Check if extension should be enabled
            enabled = self._should_enable_extension(name)

            # Load MCP servers
            mcp_servers = config.get("mcpServers", {})

            # Load context files
            context_files = []
            context_filename = config.get("contextFileName", "GEMINI.md")
            context_file = extension_dir / context_filename
            if context_file.exists():
                context_files.append(context_file)

            # Load exclude tools
            exclude_tools = config.get("excludeTools", [])

            # Load dependencies
            dependencies = config.get("dependencies", [])

            return ExtensionInfo(
                name=name,
                version=version,
                description=description,
                enabled=enabled,
                path=extension_dir,
                config=config,
                dependencies=dependencies,
                mcp_servers=mcp_servers,
                context_files=context_files,
                exclude_tools=exclude_tools
            )

        except Exception as e:
            self.logger.error(f"Failed to parse extension config {config_file}: {e}")
            return None

    def _should_enable_extension(self, name: str) -> bool:
        """Determine if an extension should be enabled."""
        # Check explicit configuration
        if self.enabled_extensions:
            return name in self.enabled_extensions

        # Check if disabled
        if name in self.disabled_extensions:
            return False

        # Default to enabled if no specific configuration
        return True

    async def _load_enabled_extensions(self):
        """Load all enabled extensions."""
        for name, ext_info in self.extensions.items():
            if not ext_info.enabled:
                continue

            try:
                await self._load_single_extension(ext_info)
            except Exception as e:
                self.logger.error(f"Failed to load extension {name}: {e}")
                continue

    async def _load_single_extension(self, ext_info: ExtensionInfo):
        """Load a single extension."""
        try:
            # Check dependencies
            if not await self._check_dependencies(ext_info):
                self.logger.warning(f"Extension {ext_info.name} has unmet dependencies")
                return

            # Look for Python module
            python_module = ext_info.path / "extension.py"
            if python_module.exists():
                extension = await self._load_python_extension(python_module, ext_info)
                if extension:
                    self.loaded_extensions[ext_info.name] = extension
                    await self._register_extension_components(extension, ext_info)

            # Register MCP servers from extension
            await self._register_extension_mcp_servers(ext_info)

            # Register context files
            await self._register_extension_context(ext_info)

            self.logger.info(f"Loaded extension: {ext_info.name}")

        except Exception as e:
            self.logger.error(f"Failed to load extension {ext_info.name}: {e}")
            raise ExtensionError(f"Extension loading failed: {e}")

    async def _load_python_extension(self, module_path: Path, ext_info: ExtensionInfo) -> Optional[BaseExtension]:
        """Load a Python extension module."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(
                f"gemini_extension_{ext_info.name}",
                module_path
            )

            if not spec or not spec.loader:
                self.logger.error(f"Could not load extension module: {module_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Look for extension class
            extension_class = getattr(module, "Extension", None)
            if not extension_class:
                self.logger.error(f"Extension {ext_info.name} missing Extension class")
                return None

            # Create extension instance
            extension = extension_class(ext_info.name, ext_info.version)

            # Initialize extension
            cli_context = {
                "config": self.config,
                "extension_config": ext_info.config,
                "extension_path": ext_info.path
            }

            if await extension.initialize(cli_context):
                return extension
            else:
                self.logger.error(f"Extension {ext_info.name} initialization failed")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load Python extension {ext_info.name}: {e}")
            return None

    async def _check_dependencies(self, ext_info: ExtensionInfo) -> bool:
        """Check if extension dependencies are met."""
        for dependency in ext_info.dependencies:
            if dependency not in self.extensions:
                self.logger.warning(f"Extension {ext_info.name} requires {dependency}")
                return False

            # Check if dependency is enabled
            if not self.extensions[dependency].enabled:
                self.logger.warning(f"Extension {ext_info.name} requires enabled {dependency}")
                return False

        return True

    async def _register_extension_components(self, extension: BaseExtension, ext_info: ExtensionInfo):
        """Register extension commands and hooks."""
        # Register commands
        commands = extension.get_commands()
        for cmd_name, cmd_func in commands.items():
            full_cmd_name = f"{ext_info.name}:{cmd_name}"
            self.extension_commands[full_cmd_name] = cmd_func
            self.logger.debug(f"Registered command: {full_cmd_name}")

        # Register hooks
        hooks = extension.get_hooks()
        for hook_name, hook_funcs in hooks.items():
            if hook_name not in self.extension_hooks:
                self.extension_hooks[hook_name] = []
            self.extension_hooks[hook_name].extend(hook_funcs)
            self.logger.debug(f"Registered {len(hook_funcs)} hooks for: {hook_name}")

    async def _register_extension_mcp_servers(self, ext_info: ExtensionInfo):
        """Register MCP servers from extension."""
        # This would integrate with the MCP system
        # For now, just log the servers
        for server_name, server_config in ext_info.mcp_servers.items():
            full_server_name = f"{ext_info.name}:{server_name}"
            self.logger.debug(f"Extension {ext_info.name} provides MCP server: {full_server_name}")

    async def _register_extension_context(self, ext_info: ExtensionInfo):
        """Register extension context files."""
        # This would integrate with the memory system
        for context_file in ext_info.context_files:
            self.logger.debug(f"Extension {ext_info.name} provides context: {context_file}")

    def list_extensions(self) -> List[ExtensionInfo]:
        """List all discovered extensions."""
        return list(self.extensions.values())

    def get_extension(self, name: str) -> Optional[ExtensionInfo]:
        """Get extension info by name."""
        return self.extensions.get(name)

    def get_loaded_extension(self, name: str) -> Optional[BaseExtension]:
        """Get loaded extension instance by name."""
        return self.loaded_extensions.get(name)

    async def enable_extension(self, name: str) -> bool:
        """Enable an extension."""
        ext_info = self.extensions.get(name)
        if not ext_info:
            self.logger.error(f"Extension not found: {name}")
            return False

        if ext_info.enabled:
            self.logger.info(f"Extension {name} already enabled")
            return True

        try:
            ext_info.enabled = True
            await self._load_single_extension(ext_info)

            # Update configuration
            self.enabled_extensions.add(name)
            self.disabled_extensions.discard(name)

            self.logger.info(f"Enabled extension: {name}")
            return True

        except Exception as e:
            ext_info.enabled = False
            self.logger.error(f"Failed to enable extension {name}: {e}")
            return False

    async def disable_extension(self, name: str) -> bool:
        """Disable an extension."""
        ext_info = self.extensions.get(name)
        if not ext_info:
            self.logger.error(f"Extension not found: {name}")
            return False

        if not ext_info.enabled:
            self.logger.info(f"Extension {name} already disabled")
            return True

        try:
            # Shutdown extension if loaded
            loaded_ext = self.loaded_extensions.get(name)
            if loaded_ext:
                await loaded_ext.shutdown()
                del self.loaded_extensions[name]

            # Remove commands and hooks
            self._unregister_extension_components(name)

            ext_info.enabled = False

            # Update configuration
            self.disabled_extensions.add(name)
            self.enabled_extensions.discard(name)

            self.logger.info(f"Disabled extension: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to disable extension {name}: {e}")
            return False

    def _unregister_extension_components(self, extension_name: str):
        """Unregister extension commands and hooks."""
        # Remove commands
        commands_to_remove = [
            cmd_name for cmd_name in self.extension_commands.keys()
            if cmd_name.startswith(f"{extension_name}:")
        ]

        for cmd_name in commands_to_remove:
            del self.extension_commands[cmd_name]

        # Remove hooks (simplified - would need more sophisticated tracking)
        # This is a basic implementation
        for hook_name, hook_list in self.extension_hooks.items():
            # Filter out hooks from this extension
            # In a real implementation, we'd need better tracking
            pass

    async def execute_extension_command(self, command: str, args: List[str]) -> Any:
        """Execute an extension command."""
        if command not in self.extension_commands:
            raise ExtensionError(f"Extension command not found: {command}")

        try:
            cmd_func = self.extension_commands[command]
            return await cmd_func(args)

        except Exception as e:
            self.logger.error(f"Extension command {command} failed: {e}")
            raise ExtensionError(f"Extension command failed: {e}")

    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger extension hooks."""
        if hook_name not in self.extension_hooks:
            return []

        results = []

        for hook_func in self.extension_hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    result = await hook_func(*args, **kwargs)
                else:
                    result = hook_func(*args, **kwargs)
                results.append(result)

            except Exception as e:
                self.logger.error(f"Hook {hook_name} failed: {e}")
                results.append(None)

        return results

    def get_extension_commands(self) -> Dict[str, str]:
        """Get all available extension commands."""
        return {
            cmd_name: f"Extension command from {cmd_name.split(':')[0]}"
            for cmd_name in self.extension_commands.keys()
        }

    def get_extension_tools(self) -> List[Dict[str, Any]]:
        """Get tools provided by extensions."""
        tools = []

        for extension in self.loaded_extensions.values():
            ext_tools = extension.get_tools()
            for tool in ext_tools:
                tool["extension"] = extension.name
                tools.append(tool)

        return tools

    def get_extension_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP servers provided by extensions."""
        servers = {}

        for ext_info in self.extensions.values():
            if not ext_info.enabled:
                continue

            for server_name, server_config in ext_info.mcp_servers.items():
                full_name = f"{ext_info.name}:{server_name}"
                servers[full_name] = {
                    **server_config,
                    "extension": ext_info.name,
                    "extension_path": str(ext_info.path)
                }

        return servers

    def get_extension_context_files(self) -> List[Path]:
        """Get context files provided by extensions."""
        context_files = []

        for ext_info in self.extensions.values():
            if not ext_info.enabled:
                continue

            context_files.extend(ext_info.context_files)

        return context_files

    def get_extension_exclude_tools(self) -> List[str]:
        """Get tools to exclude based on extensions."""
        exclude_tools = []

        for ext_info in self.extensions.values():
            if not ext_info.enabled:
                continue

            exclude_tools.extend(ext_info.exclude_tools)

        return exclude_tools

    async def shutdown_all_extensions(self):
        """Shutdown all loaded extensions."""
        self.logger.info("Shutting down extensions...")

        for name, extension in self.loaded_extensions.items():
            try:
                await extension.shutdown()
                self.logger.debug(f"Shutdown extension: {name}")
            except Exception as e:
                self.logger.error(f"Failed to shutdown extension {name}: {e}")

        self.loaded_extensions.clear()
        self.extension_commands.clear()
        self.extension_hooks.clear()

    def get_extension_stats(self) -> Dict[str, Any]:
        """Get extension statistics."""
        return {
            "total_extensions": len(self.extensions),
            "enabled_extensions": len([e for e in self.extensions.values() if e.enabled]),
            "loaded_extensions": len(self.loaded_extensions),
            "extension_commands": len(self.extension_commands),
            "extension_hooks": sum(len(hooks) for hooks in self.extension_hooks.values()),
            "extensions_by_source": {
                "workspace": len([e for e in self.extensions.values() if "workspace" in str(e.path)]),
                "user": len([e for e in self.extensions.values() if "user" in str(e.path)])
            }
        }

    async def install_extension(self, source: str, target_dir: Optional[Path] = None) -> bool:
        """Install an extension from a source."""
        # This would implement extension installation
        # For now, it's a placeholder
        self.logger.info(f"Extension installation not yet implemented: {source}")
        return False

    async def uninstall_extension(self, name: str) -> bool:
        """Uninstall an extension."""
        ext_info = self.extensions.get(name)
        if not ext_info:
            self.logger.error(f"Extension not found: {name}")
            return False

        try:
            # Disable extension first
            await self.disable_extension(name)

            # Remove extension directory (with confirmation)
            import shutil
            shutil.rmtree(ext_info.path)

            # Remove from registry
            del self.extensions[name]

            self.logger.info(f"Uninstalled extension: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to uninstall extension {name}: {e}")
            return False

    async def reload_extension(self, name: str) -> bool:
        """Reload an extension."""
        try:
            # Disable and re-enable
            await self.disable_extension(name)

            # Re-discover the extension
            ext_info = self.extensions.get(name)
            if ext_info:
                config_file = ext_info.path / "gemini-extension.json"
                if config_file.exists():
                    source = "workspace" if "workspace" in str(ext_info.path) else "user"
                    new_ext_info = await self._load_extension_config(config_file, ext_info.path, source)
                    if new_ext_info:
                        self.extensions[name] = new_ext_info

            await self.enable_extension(name)

            self.logger.info(f"Reloaded extension: {name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reload extension {name}: {e}")
            return False


# Example extension for reference
class ExampleExtension(BaseExtension):
    """Example extension implementation."""

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.description = "Example extension for Gemini CLI"
        self.author = "Gemini CLI Team"

    async def initialize(self, cli_context: Dict[str, Any]) -> bool:
        """Initialize the example extension."""
        self.logger.info("Example extension initialized")
        return True

    async def shutdown(self) -> bool:
        """Shutdown the example extension."""
        self.logger.info("Example extension shutdown")
        return True

    def get_commands(self) -> Dict[str, Callable]:
        """Get extension commands."""
        return {
            "hello": self._hello_command,
            "info": self._info_command
        }

    async def _hello_command(self, args: List[str]) -> str:
        """Example hello command."""
        name = args[0] if args else "World"
        return f"Hello, {name}! From {self.name} extension."

    async def _info_command(self, args: List[str]) -> str:
        """Example info command."""
        return f"Extension: {self.name} v{self.version}\nDescription: {self.description}"

    def get_hooks(self) -> Dict[str, List[Callable]]:
        """Get extension hooks."""
        return {
            "before_prompt": [self._before_prompt_hook],
            "after_response": [self._after_response_hook]
        }

    async def _before_prompt_hook(self, prompt: str, context: Dict[str, Any]) -> None:
        """Example before prompt hook."""
        self.logger.debug(f"Before prompt: {prompt[:50]}...")

    async def _after_response_hook(self, response: str, context: Dict[str, Any]) -> None:
        """Example after response hook."""
        self.logger.debug(f"After response: {len(response)} characters")