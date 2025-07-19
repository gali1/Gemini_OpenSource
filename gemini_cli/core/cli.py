"""
Core CLI implementation for Gemini OpenSource
"""

import asyncio
import sys
import os
import readline
import atexit
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import json
import shlex
import signal
import time
from datetime import datetime

import torch

from .config import ConfigManager
from .session import SessionManager
from .display import DisplayManager
from .memory import MemoryManager
from .tools import ToolRegistry
from .extensions import ExtensionManager
from .file_discovery import FileDiscoveryService
from .checkpointing import CheckpointManager
from .telemetry import TelemetryManager
from .sandbox import SandboxManager
from .exceptions import GeminiCLIError, ToolExecutionError, ModelError
from ..model_interface import GeminiModelInterface
from ..utils.shell import ShellMode
from ..utils.history import HistoryManager
from ..utils.theme import ThemeManager
from ..utils.stats import StatsTracker


class GeminiCLI:
    """Main CLI class that orchestrates all components."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()

        # Core components
        self.session_manager = SessionManager()
        self.display = DisplayManager(self.config.get("theme", "Default"))
        self.memory_manager = MemoryManager(self.config)
        self.tool_registry = ToolRegistry(self.config)
        self.extension_manager = ExtensionManager(self.config)
        self.file_discovery = FileDiscoveryService(self.config)
        self.checkpoint_manager = CheckpointManager(self.config) if self.config.get("checkpointing", {}).get("enabled") else None
        self.telemetry_manager = TelemetryManager(self.config) if self.config.get("telemetry", {}).get("enabled") else None
        self.sandbox_manager = SandboxManager(self.config) if self.config.get("sandbox", False) else None

        # Model interface
        self.model_interface = GeminiModelInterface(self.config)

        # Utilities
        self.shell_mode = ShellMode()
        self.history_manager = HistoryManager()
        self.theme_manager = ThemeManager()
        self.stats_tracker = StatsTracker()

        # State
        self.running = False
        self.session_id = None
        self.current_conversation = []
        self.context_files = []

        # Setup readline for better CLI experience
        self._setup_readline()

    def _setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            # Load history
            history_file = Path.home() / ".gemini" / "cli_history"
            history_file.parent.mkdir(exist_ok=True)

            if history_file.exists():
                readline.read_history_file(str(history_file))

            # Set history length
            readline.set_history_length(1000)

            # Save history on exit
            atexit.register(lambda: readline.write_history_file(str(history_file)))

            # Setup tab completion
            readline.set_completer(self._completer)
            readline.parse_and_bind("tab: complete")

        except Exception as e:
            # Readline might not be available on all systems
            print(f"Warning: Could not setup readline: {e}")

    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion handler."""
        if state == 0:
            # Initialize completion list
            self._completion_matches = []

            # Complete commands starting with /
            if text.startswith("/"):
                commands = [
                    "/help", "/quit", "/exit", "/clear", "/stats", "/theme", "/tools",
                    "/memory", "/mcp", "/chat", "/compress", "/editor", "/restore",
                    "/auth", "/about", "/privacy", "/bug", "/extensions"
                ]
                self._completion_matches = [cmd for cmd in commands if cmd.startswith(text)]

            # Complete file paths starting with @
            elif text.startswith("@"):
                try:
                    path_part = text[1:]  # Remove @
                    if "/" in path_part:
                        directory = str(Path(path_part).parent)
                        filename = Path(path_part).name
                    else:
                        directory = "."
                        filename = path_part

                    try:
                        files = os.listdir(directory)
                        matches = [f for f in files if f.startswith(filename)]
                        self._completion_matches = [f"@{directory}/{match}" if directory != "." else f"@{match}" for match in matches]
                    except OSError:
                        self._completion_matches = []
                except Exception:
                    self._completion_matches = []

            # Complete shell commands starting with !
            elif text.startswith("!"):
                # Simple shell command completion - could be enhanced
                common_commands = ["!ls", "!pwd", "!cd", "!grep", "!find", "!git", "!npm", "!python"]
                self._completion_matches = [cmd for cmd in common_commands if cmd.startswith(text)]

        try:
            return self._completion_matches[state]
        except IndexError:
            return None

    async def initialize(self):
        """Initialize all CLI components."""
        self.display.show_status("Initializing Gemini CLI...")

        try:
            # Initialize telemetry first
            if self.telemetry_manager:
                await self.telemetry_manager.initialize()

            # Initialize model interface
            await self.model_interface.initialize()

            # Initialize tools
            await self.tool_registry.initialize()

            # Load extensions
            await self.extension_manager.load_extensions()

            # Initialize memory
            await self.memory_manager.initialize()

            # Start new session
            self.session_id = self.session_manager.create_session()

            # Initialize sandbox if enabled
            if self.sandbox_manager:
                await self.sandbox_manager.initialize()

            self.display.show_success("Gemini CLI initialized successfully!")

        except Exception as e:
            self.display.show_error(f"Failed to initialize CLI: {e}")
            raise GeminiCLIError(f"Initialization failed: {e}")

    async def run_interactive(self) -> int:
        """Run the CLI in interactive mode."""
        self.running = True

        try:
            self.display.show_welcome()
            self._show_context_info()

            while self.running:
                try:
                    # Get user input
                    prompt = self._get_user_input()

                    if prompt is None:  # EOF or quit
                        break

                    if not prompt.strip():
                        continue

                    # Process the input
                    await self._process_input(prompt)

                except KeyboardInterrupt:
                    print("\n(Use /quit or Ctrl+D to exit)")
                    continue
                except EOFError:
                    break
                except Exception as e:
                    self.display.show_error(f"Error processing input: {e}")
                    if self.config.get("debug", False):
                        import traceback
                        traceback.print_exc()

            return 0

        finally:
            await self._cleanup()

    def _get_user_input(self) -> Optional[str]:
        """Get user input with proper prompt."""
        try:
            # Check if we're in shell mode
            if self.shell_mode.active:
                prompt_text = self.display.get_shell_prompt()
            else:
                prompt_text = self.display.get_prompt()

            user_input = input(prompt_text)

            # Handle shell mode toggle
            if user_input.strip() == "!":
                self.shell_mode.toggle()
                self.display.show_info(f"Shell mode {'activated' if self.shell_mode.active else 'deactivated'}")
                return ""

            return user_input

        except EOFError:
            return None
        except KeyboardInterrupt:
            raise

    async def _process_input(self, user_input: str):
        """Process user input and route to appropriate handler."""
        user_input = user_input.strip()

        if not user_input:
            return

        # Track the interaction
        self.stats_tracker.record_user_prompt(user_input)

        # Handle shell mode
        if self.shell_mode.active and not user_input.startswith(("/", "@")):
            await self._handle_shell_command(user_input)
            return

        # Handle commands starting with special characters
        if user_input.startswith("/"):
            await self._handle_slash_command(user_input)
        elif user_input.startswith("@"):
            await self._handle_at_command(user_input)
        elif user_input.startswith("!"):
            await self._handle_shell_command(user_input[1:])
        else:
            # Regular AI prompt
            await self._handle_ai_prompt(user_input)

    async def _handle_slash_command(self, command: str):
        """Handle slash commands."""
        parts = shlex.split(command[1:])  # Remove / and parse
        cmd = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        try:
            if cmd in ["quit", "exit"]:
                self.display.show_info("Goodbye!")
                self.running = False

            elif cmd == "help" or cmd == "?":
                self._show_help(args)

            elif cmd == "clear":
                os.system('clear' if os.name == 'posix' else 'cls')

            elif cmd == "stats":
                self._show_stats()

            elif cmd == "theme":
                await self._handle_theme_command(args)

            elif cmd == "tools":
                await self._handle_tools_command(args)

            elif cmd == "memory":
                await self._handle_memory_command(args)

            elif cmd == "mcp":
                await self._handle_mcp_command(args)

            elif cmd == "chat":
                await self._handle_chat_command(args)

            elif cmd == "compress":
                await self._handle_compress_command()

            elif cmd == "editor":
                await self._handle_editor_command()

            elif cmd == "restore":
                await self._handle_restore_command(args)

            elif cmd == "auth":
                await self._handle_auth_command()

            elif cmd == "about":
                self._show_about()

            elif cmd == "privacy":
                self._show_privacy()

            elif cmd == "bug":
                await self._handle_bug_command(args)

            elif cmd == "extensions":
                await self._handle_extensions_command(args)

            else:
                self.display.show_error(f"Unknown command: /{cmd}. Type /help for available commands.")

        except Exception as e:
            self.display.show_error(f"Error executing command /{cmd}: {e}")

    async def _handle_at_command(self, command: str):
        """Handle @ file inclusion commands."""
        try:
            # Parse the file path
            if command.strip() == "@":
                self.display.show_error("Usage: @<file_path> [your prompt]")
                return

            # Extract file path and optional prompt
            parts = command[1:].split(None, 1)
            file_path = parts[0]
            additional_prompt = parts[1] if len(parts) > 1 else ""

            # Resolve and read files
            files = self.file_discovery.discover_files(file_path)

            if not files:
                self.display.show_error(f"No files found matching: {file_path}")
                return

            # Read file contents
            file_contents = []
            for file in files:
                try:
                    content = await self.tool_registry.execute_tool("read_file", {"path": str(file)})
                    file_contents.append(f"--- {file} ---\n{content}")
                except Exception as e:
                    self.display.show_warning(f"Could not read {file}: {e}")

            if not file_contents:
                self.display.show_error("No files could be read")
                return

            # Combine with prompt
            full_prompt = "\n\n".join(file_contents)
            if additional_prompt:
                full_prompt += f"\n\nUser question: {additional_prompt}"

            # Process as AI prompt
            await self._handle_ai_prompt(full_prompt, context_files=files)

        except Exception as e:
            self.display.show_error(f"Error processing @ command: {e}")

    async def _handle_shell_command(self, command: str):
        """Handle shell commands."""
        try:
            if self.sandbox_manager:
                result = await self.sandbox_manager.execute_command(command)
            else:
                result = await self.tool_registry.execute_tool("run_shell_command", {"command": command})

            self.display.show_shell_result(result)

        except Exception as e:
            self.display.show_error(f"Error executing shell command: {e}")

    async def _handle_ai_prompt(self, prompt: str, context_files: Optional[List[Path]] = None):
        """Handle AI prompts."""
        try:
            self.display.show_thinking()

            # Add context files to conversation context
            context = []
            if context_files:
                for file in context_files:
                    try:
                        content = await self.tool_registry.execute_tool("read_file", {"path": str(file)})
                        context.append(f"File: {file}\n{content}")
                    except Exception as e:
                        self.display.show_warning(f"Could not include {file}: {e}")

            # Get memory context
            memory_context = await self.memory_manager.get_context()
            if memory_context:
                context.append(f"Memory Context:\n{memory_context}")

            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = "\n\n".join(context) + f"\n\nUser: {prompt}"

            # Get model response
            response = await self.model_interface.generate_response(
                prompt=full_prompt,
                conversation_history=self.current_conversation,
                available_tools=self.tool_registry.get_tool_schemas()
            )

            # Process any tool calls in the response
            if response.get("tool_calls"):
                response = await self._handle_tool_calls(response, prompt)

            # Display response
            self.display.show_response(response)

            # Update conversation history
            self.current_conversation.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            self.current_conversation.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "timestamp": datetime.now().isoformat()
            })

            # Track stats
            self.stats_tracker.record_api_request(response.get("usage", {}))

        except Exception as e:
            self.display.show_error(f"Error processing AI prompt: {e}")
            if self.config.get("debug", False):
                import traceback
                traceback.print_exc()

    async def _handle_tool_calls(self, response: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Handle tool calls from the model."""
        tool_calls = response.get("tool_calls", [])

        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})

                # Check if tool exists
                if not self.tool_registry.has_tool(tool_name):
                    self.display.show_error(f"Unknown tool: {tool_name}")
                    continue

                # Show tool call information
                self.display.show_tool_call(tool_name, tool_args)

                # Check if confirmation is needed
                if self._needs_confirmation(tool_name, tool_args):
                    if not self._confirm_tool_execution(tool_name, tool_args):
                        self.display.show_info("Tool execution cancelled by user.")
                        continue

                # Create checkpoint if enabled
                if self.checkpoint_manager:
                    checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                        tool_name, tool_args, original_prompt
                    )
                    self.display.show_info(f"Checkpoint created: {checkpoint_id}")

                # Execute tool
                self.display.show_status(f"Executing {tool_name}...")
                start_time = time.time()

                try:
                    result = await self.tool_registry.execute_tool(tool_name, tool_args)
                    execution_time = time.time() - start_time

                    self.display.show_tool_result(tool_name, result, execution_time)
                    self.stats_tracker.record_tool_call(tool_name, True, execution_time)

                except ToolExecutionError as e:
                    execution_time = time.time() - start_time
                    self.display.show_error(f"Tool execution failed: {e}")
                    self.stats_tracker.record_tool_call(tool_name, False, execution_time)
                    continue

                # Update response with tool result
                # The model would typically generate a follow-up response
                # incorporating the tool results

            except Exception as e:
                self.display.show_error(f"Error handling tool call: {e}")

        return response

    def _needs_confirmation(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Check if tool execution needs user confirmation."""
        # Auto-accept mode
        if self.config.get("autoAccept", False):
            return False

        # Check tool-specific settings
        dangerous_tools = ["run_shell_command", "write_file", "replace"]
        return tool_name in dangerous_tools

    def _confirm_tool_execution(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        """Ask user to confirm tool execution."""
        try:
            self.display.show_tool_confirmation(tool_name, tool_args)

            while True:
                response = input("Proceed? (y/n/m for modify): ").strip().lower()

                if response in ["y", "yes"]:
                    return True
                elif response in ["n", "no"]:
                    return False
                elif response in ["m", "modify"]:
                    # Could implement tool argument modification
                    self.display.show_info("Tool modification not yet implemented.")
                    return False
                else:
                    print("Please answer 'y' (yes), 'n' (no), or 'm' (modify)")

        except KeyboardInterrupt:
            return False

    def _show_help(self, args: List[str]):
        """Show help information."""
        if not args:
            help_text = """
Gemini CLI - Interactive AI Assistant

SLASH COMMANDS:
  /help, /?          Show this help message
  /quit, /exit       Exit the CLI
  /clear             Clear the screen
  /stats             Show session statistics
  /theme             Change color theme
  /tools [desc]      List available tools
  /memory            Manage AI memory
  /mcp               Show MCP server status
  /chat              Manage conversation state
  /compress          Compress conversation history
  /editor            Select preferred editor
  /restore [id]      Restore from checkpoint
  /auth              Authentication settings
  /about             Show version information
  /privacy           Show privacy information
  /bug [title]       Report a bug
  /extensions        Manage extensions

AT COMMANDS:
  @<file_path>       Include file content in prompt
  @<pattern>         Include multiple files matching pattern

SHELL COMMANDS:
  !<command>         Execute shell command
  !                  Toggle shell mode

EXAMPLES:
  @src/main.py Explain this code
  !git status
  /memory add "Project uses FastAPI framework"
  /stats
"""
            print(help_text)
        else:
            # Show help for specific command
            command = args[0]
            # Implementation for specific command help
            self.display.show_info(f"Help for /{command} - Documentation not yet implemented")

    def _show_stats(self):
        """Show session statistics."""
        stats = self.stats_tracker.get_stats()
        self.display.show_stats(stats)

    def _show_context_info(self):
        """Show information about loaded context."""
        memory_files = self.memory_manager.get_loaded_files()
        if memory_files:
            self.display.show_info(f"Loaded {len(memory_files)} context file(s)")

    def _show_about(self):
        """Show about information."""
        from ..utils.version import get_version
        print(f"""
Gemini CLI {get_version()}
Interactive AI Assistant powered by Gemini OpenSource

Session ID: {self.session_id}
Model: {self.config.get('model', 'gemini-torch')}
Theme: {self.config.get('theme', 'Default')}
Sandbox: {'Enabled' if self.sandbox_manager else 'Disabled'}

For more information, visit:
https://github.com/kyegomez/Gemini
        """)

    def _show_privacy(self):
        """Show privacy information."""
        print("""
Gemini CLI Privacy Information

This is an open-source implementation of Gemini that runs locally.
- No data is sent to external servers unless explicitly configured
- Conversation history is stored locally
- Tool executions are performed on your local system
- Telemetry is optional and can be disabled

For full privacy details, see:
https://github.com/kyegomez/Gemini/blob/main/PRIVACY.md
        """)

    async def _handle_theme_command(self, args: List[str]):
        """Handle theme command."""
        if not args:
            # Show theme selector
            themes = self.theme_manager.list_themes()
            self.display.show_theme_selector(themes)
        else:
            theme_name = args[0]
            if self.theme_manager.set_theme(theme_name):
                self.display.set_theme(theme_name)
                self.config_manager.update_config({"theme": theme_name})
                self.display.show_success(f"Theme changed to: {theme_name}")
            else:
                self.display.show_error(f"Unknown theme: {theme_name}")

    async def _handle_tools_command(self, args: List[str]):
        """Handle tools command."""
        show_descriptions = "desc" in args or "descriptions" in args
        tools = self.tool_registry.list_tools(include_descriptions=show_descriptions)
        self.display.show_tools(tools, show_descriptions)

    async def _handle_memory_command(self, args: List[str]):
        """Handle memory commands."""
        if not args:
            self.display.show_error("Memory command requires an action: show, add, refresh")
            return

        action = args[0]

        if action == "show":
            context = await self.memory_manager.get_context()
            self.display.show_memory_context(context)

        elif action == "add":
            if len(args) < 2:
                self.display.show_error("Memory add requires text to remember")
                return

            text = " ".join(args[1:])
            await self.memory_manager.add_memory(text)
            self.display.show_success("Memory added successfully")

        elif action == "refresh":
            await self.memory_manager.refresh()
            self.display.show_success("Memory refreshed from files")

        else:
            self.display.show_error(f"Unknown memory action: {action}")

    async def _handle_mcp_command(self, args: List[str]):
        """Handle MCP server commands."""
        show_descriptions = "desc" in args or "descriptions" in args
        servers = self.tool_registry.get_mcp_servers()
        self.display.show_mcp_servers(servers, show_descriptions)

    async def _handle_chat_command(self, args: List[str]):
        """Handle chat state management."""
        if not args:
            self.display.show_error("Chat command requires an action: save, resume, list")
            return

        action = args[0]

        if action == "save":
            if len(args) < 2:
                self.display.show_error("Chat save requires a tag name")
                return

            tag = args[1]
            self.session_manager.save_conversation(self.current_conversation, tag)
            self.display.show_success(f"Conversation saved as: {tag}")

        elif action == "resume":
            if len(args) < 2:
                self.display.show_error("Chat resume requires a tag name")
                return

            tag = args[1]
            conversation = self.session_manager.load_conversation(tag)
            if conversation:
                self.current_conversation = conversation
                self.display.show_success(f"Conversation resumed: {tag}")
            else:
                self.display.show_error(f"No saved conversation found: {tag}")

        elif action == "list":
            tags = self.session_manager.list_saved_conversations()
            self.display.show_saved_conversations(tags)

        else:
            self.display.show_error(f"Unknown chat action: {action}")

    async def _handle_compress_command(self):
        """Handle conversation compression."""
        if not self.current_conversation:
            self.display.show_info("No conversation to compress")
            return

        # Implement conversation compression
        self.display.show_info("Compressing conversation history...")

        # This would use the model to create a summary
        compressed = await self.model_interface.compress_conversation(self.current_conversation)

        self.current_conversation = compressed
        self.display.show_success("Conversation history compressed")

    async def _handle_editor_command(self):
        """Handle editor selection."""
        editors = ["vscode", "vim", "nano", "emacs", "sublime"]
        self.display.show_editor_selector(editors)

    async def _handle_restore_command(self, args: List[str]):
        """Handle checkpoint restoration."""
        if not self.checkpoint_manager:
            self.display.show_error("Checkpointing is not enabled")
            return

        if not args:
            # List available checkpoints
            checkpoints = await self.checkpoint_manager.list_checkpoints()
            self.display.show_checkpoints(checkpoints)
        else:
            # Restore specific checkpoint
            checkpoint_id = args[0]
            try:
                await self.checkpoint_manager.restore_checkpoint(checkpoint_id)
                self.display.show_success(f"Restored checkpoint: {checkpoint_id}")
            except Exception as e:
                self.display.show_error(f"Failed to restore checkpoint: {e}")

    async def _handle_auth_command(self):
        """Handle authentication settings."""
        self.display.show_auth_info(self.config.get("auth", {}))

    async def _handle_bug_command(self, args: List[str]):
        """Handle bug reporting."""
        title = " ".join(args) if args else "Bug Report"
        bug_url = self.config.get("bugCommand", {}).get("urlTemplate",
                                  "https://github.com/kyegomez/Gemini/issues/new")

        # Replace placeholders in URL
        bug_url = bug_url.replace("{title}", title)
        bug_url = bug_url.replace("{info}", f"CLI Version: {self.model_interface.get_version()}")

        self.display.show_info(f"Opening bug report: {bug_url}")

        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(bug_url)
        except Exception:
            self.display.show_info(f"Please visit: {bug_url}")

    async def _handle_extensions_command(self, args: List[str]):
        """Handle extension management."""
        extensions = self.extension_manager.list_extensions()
        self.display.show_extensions(extensions)

    async def execute_prompt(self, prompt: str, context_files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Execute a single prompt (for non-interactive mode)."""
        try:
            # Prepare context
            context = []
            if context_files:
                for file in context_files:
                    try:
                        content = await self.tool_registry.execute_tool("read_file", {"path": str(file)})
                        context.append(f"File: {file}\n{content}")
                    except Exception as e:
                        self.display.show_warning(f"Could not include {file}: {e}")

            # Get memory context
            memory_context = await self.memory_manager.get_context()
            if memory_context:
                context.append(f"Memory Context:\n{memory_context}")

            # Prepare full prompt
            full_prompt = prompt
            if context:
                full_prompt = "\n\n".join(context) + f"\n\nUser: {prompt}"

            # Get model response
            response = await self.model_interface.generate_response(
                prompt=full_prompt,
                conversation_history=[],  # No history in single-prompt mode
                available_tools=self.tool_registry.get_tool_schemas()
            )

            # Handle tool calls if any
            if response.get("tool_calls"):
                response = await self._handle_tool_calls(response, prompt)

            return response

        except Exception as e:
            raise GeminiCLIError(f"Error executing prompt: {e}")

    def shutdown(self):
        """Shutdown the CLI gracefully."""
        self.running = False
        print("\nShutting down...")

    async def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            # Save session
            if self.session_id and self.current_conversation:
                self.session_manager.save_session(self.session_id, self.current_conversation)

            # Cleanup sandbox
            if self.sandbox_manager:
                await self.sandbox_manager.cleanup()

            # Show final stats
            if self.stats_tracker.has_activity():
                self.display.show_final_stats(self.stats_tracker.get_stats())

            # Shutdown telemetry
            if self.telemetry_manager:
                await self.telemetry_manager.shutdown()

            print("Goodbye!")

        except Exception as e:
            print(f"Error during cleanup: {e}")