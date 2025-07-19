"""
Display manager for Gemini CLI
Handles all user interface, formatting, and output display
"""

import os
import sys
import json
import time
import psutil
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.text import Text
    from rich.tree import Tree
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..utils.colors import Colors, ColorTheme
from ..utils.formatting import format_file_size, format_duration, format_timestamp


class DisplayManager:
    """Manages all display and user interface operations."""

    def __init__(self, theme: str = "Default"):
        self.theme_name = theme
        self.colors = Colors()
        self.color_theme = ColorTheme(theme)

        # Initialize rich console if available
        if RICH_AVAILABLE:
            self.console = Console(
                force_terminal=True,
                color_system="auto",
                width=None,
                legacy_windows=False
            )
            self.use_rich = True
        else:
            self.console = None
            self.use_rich = False

        # Check if NO_COLOR is set
        if os.environ.get("NO_COLOR"):
            self.use_colors = False
        else:
            self.use_colors = True

    def set_theme(self, theme_name: str):
        """Set the display theme."""
        self.theme_name = theme_name
        self.color_theme = ColorTheme(theme_name)

    def show_welcome(self):
        """Show welcome message."""
        if self.use_rich:
            welcome_panel = Panel.fit(
                "[bold blue]Welcome to Gemini CLI[/bold blue]\n"
                "Your AI-powered command line assistant\n\n"
                "Type your message or use commands like:\n"
                "• [cyan]/help[/cyan] - Show available commands\n"
                "• [cyan]@file.py[/cyan] - Include file content\n"
                "• [cyan]!command[/cyan] - Execute shell commands\n"
                "• [cyan]/quit[/cyan] - Exit the CLI",
                title="Gemini CLI",
                border_style="blue"
            )
            self.console.print(welcome_panel)
        else:
            print(self._colorize("\n" + "="*60, "blue"))
            print(self._colorize("  Welcome to Gemini CLI", "bold_blue"))
            print(self._colorize("  Your AI-powered command line assistant", "blue"))
            print(self._colorize("="*60, "blue"))
            print("\nType your message or use commands like:")
            print("  /help - Show available commands")
            print("  @file.py - Include file content")
            print("  !command - Execute shell commands")
            print("  /quit - Exit the CLI\n")

    def get_prompt(self) -> str:
        """Get the main input prompt."""
        if self.use_colors:
            return self._colorize("❯ ", "green")
        return "> "

    def get_shell_prompt(self) -> str:
        """Get the shell mode prompt."""
        if self.use_colors:
            return self._colorize("shell❯ ", "yellow")
        return "shell> "

    def show_thinking(self):
        """Show thinking indicator."""
        if self.use_rich:
            with self.console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                time.sleep(0.1)  # Brief pause to show spinner
        else:
            print(self._colorize("Thinking...", "cyan"))

    def show_status(self, message: str):
        """Show status message."""
        if self.use_rich:
            self.console.print(f"[yellow]Status:[/yellow] {message}")
        else:
            print(self._colorize(f"Status: {message}", "yellow"))

    def show_info(self, message: str):
        """Show info message."""
        if self.use_rich:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            print(self._colorize(f"Info: {message}", "blue"))

    def show_success(self, message: str):
        """Show success message."""
        if self.use_rich:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            print(self._colorize(f"✓ {message}", "green"))

    def show_warning(self, message: str):
        """Show warning message."""
        if self.use_rich:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            print(self._colorize(f"Warning: {message}", "yellow"))

    def show_error(self, message: str):
        """Show error message."""
        if self.use_rich:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            print(self._colorize(f"Error: {message}", "red"), file=sys.stderr)

    def print_response(self, response: Dict[str, Any]):
        """Print AI response."""
        content = response.get("content", "")

        if self.use_rich:
            # Format as markdown if it looks like markdown
            if self._looks_like_markdown(content):
                md = Markdown(content)
                self.console.print(md)
            else:
                self.console.print(content)
        else:
            print(content)

        # Show usage info if available
        usage = response.get("usage")
        if usage:
            self._show_usage_info(usage)

    def show_response(self, response: Dict[str, Any]):
        """Show formatted AI response."""
        self.print_response(response)

    def show_tool_call(self, tool_name: str, tool_args: Dict[str, Any]):
        """Show tool call information."""
        if self.use_rich:
            args_text = json.dumps(tool_args, indent=2)
            syntax = Syntax(args_text, "json", theme="monokai")

            panel = Panel(
                syntax,
                title=f"[cyan]Tool Call: {tool_name}[/cyan]",
                border_style="cyan"
            )
            self.console.print(panel)
        else:
            print(self._colorize(f"\n🔧 Tool Call: {tool_name}", "cyan"))
            for key, value in tool_args.items():
                print(f"  {key}: {value}")

    def show_tool_confirmation(self, tool_name: str, tool_args: Dict[str, Any]):
        """Show tool confirmation prompt."""
        self.show_tool_call(tool_name, tool_args)

        if self.use_rich:
            self.console.print("\n[yellow]This tool will be executed with the above parameters.[/yellow]")
        else:
            print(self._colorize("\nThis tool will be executed with the above parameters.", "yellow"))

    def show_tool_result(self, tool_name: str, result: Any, execution_time: float):
        """Show tool execution result."""
        if self.use_rich:
            time_str = f"[dim]({execution_time:.2f}s)[/dim]"
            self.console.print(f"[green]✓[/green] {tool_name} completed {time_str}")

            if isinstance(result, dict):
                # Format structured output
                self._show_structured_result(result)
            elif isinstance(result, str) and result:
                # Show text result
                if len(result) > 500:
                    self.console.print(f"[dim]{result[:500]}...[/dim]")
                else:
                    self.console.print(result)
        else:
            print(self._colorize(f"✓ {tool_name} completed ({execution_time:.2f}s)", "green"))
            if result:
                print(str(result)[:500] + ("..." if len(str(result)) > 500 else ""))

    def show_shell_result(self, result: Dict[str, Any]):
        """Show shell command result."""
        if self.use_rich:
            command = result.get("command", "")
            exit_code = result.get("exit_code", 0)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            # Show command and exit code
            status_color = "green" if exit_code == 0 else "red"
            self.console.print(f"[{status_color}]Command:[/{status_color}] {command}")
            self.console.print(f"[{status_color}]Exit Code:[/{status_color}] {exit_code}")

            # Show output
            if stdout:
                self.console.print("[bold]Output:[/bold]")
                self.console.print(stdout)

            if stderr:
                self.console.print("[red]Error Output:[/red]")
                self.console.print(stderr)
        else:
            command = result.get("command", "")
            exit_code = result.get("exit_code", 0)
            print(f"Command: {command}")
            print(f"Exit Code: {exit_code}")

            if result.get("stdout"):
                print("Output:")
                print(result["stdout"])

            if result.get("stderr"):
                print("Error Output:")
                print(result["stderr"])

    def show_stats(self, stats: Dict[str, Any]):
        """Show session statistics."""
        if self.use_rich:
            table = Table(title="Session Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            # Add basic stats
            table.add_row("Session Duration", format_duration(stats.get("session_duration", 0)))
            table.add_row("Total Prompts", str(stats.get("total_prompts", 0)))
            table.add_row("Total API Calls", str(stats.get("total_api_calls", 0)))
            table.add_row("Total Tool Calls", str(stats.get("total_tool_calls", 0)))

            # Token usage
            if "token_usage" in stats:
                usage = stats["token_usage"]
                table.add_row("Input Tokens", str(usage.get("input_tokens", 0)))
                table.add_row("Output Tokens", str(usage.get("output_tokens", 0)))
                table.add_row("Total Tokens", str(usage.get("total_tokens", 0)))

            self.console.print(table)
        else:
            print("\nSession Statistics:")
            print("=" * 40)
            print(f"Session Duration: {format_duration(stats.get('session_duration', 0))}")
            print(f"Total Prompts: {stats.get('total_prompts', 0)}")
            print(f"Total API Calls: {stats.get('total_api_calls', 0)}")
            print(f"Total Tool Calls: {stats.get('total_tool_calls', 0)}")

            if "token_usage" in stats:
                usage = stats["token_usage"]
                print(f"Input Tokens: {usage.get('input_tokens', 0)}")
                print(f"Output Tokens: {usage.get('output_tokens', 0)}")
                print(f"Total Tokens: {usage.get('total_tokens', 0)}")

    def show_final_stats(self, stats: Dict[str, Any]):
        """Show final statistics on exit."""
        if self.use_rich:
            panel = Panel(
                self._format_final_stats(stats),
                title="[blue]Session Summary[/blue]",
                border_style="blue"
            )
            self.console.print(panel)
        else:
            print("\nSession Summary:")
            print("=" * 40)
            print(self._format_final_stats_plain(stats))

    def show_memory_usage(self):
        """Show current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            if self.use_rich:
                self.console.print(f"[blue]Memory Usage:[/blue] {memory_mb:.1f} MB")
            else:
                print(f"Memory Usage: {memory_mb:.1f} MB")
        except Exception:
            self.show_warning("Could not retrieve memory usage")

    def show_tools(self, tools: List[Dict[str, Any]], show_descriptions: bool = False):
        """Show available tools."""
        if self.use_rich:
            if show_descriptions:
                for tool in tools:
                    panel = Panel(
                        tool.get("description", "No description"),
                        title=f"[cyan]{tool['name']}[/cyan]",
                        border_style="cyan"
                    )
                    self.console.print(panel)
            else:
                table = Table(title="Available Tools")
                table.add_column("Tool Name", style="cyan")
                table.add_column("Description", style="white")

                for tool in tools:
                    table.add_row(tool["name"], tool.get("description", ""))

                self.console.print(table)
        else:
            print("\nAvailable Tools:")
            print("=" * 40)
            for tool in tools:
                print(f"• {tool['name']}")
                if show_descriptions:
                    print(f"  {tool.get('description', '')}")

    def show_memory_context(self, context: str):
        """Show memory context."""
        if self.use_rich:
            if context:
                md = Markdown(context)
                panel = Panel(md, title="[green]Memory Context[/green]", border_style="green")
                self.console.print(panel)
            else:
                self.console.print("[yellow]No memory context loaded[/yellow]")
        else:
            print("\nMemory Context:")
            print("=" * 40)
            if context:
                print(context)
            else:
                print("No memory context loaded")

    def show_mcp_servers(self, servers: List[Dict[str, Any]], show_descriptions: bool = False):
        """Show MCP server status."""
        if self.use_rich:
            table = Table(title="MCP Servers")
            table.add_column("Server", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Tools", style="white")

            for server in servers:
                status_color = "green" if server.get("connected") else "red"
                status_text = f"[{status_color}]{server.get('status', 'Unknown')}[/{status_color}]"
                tools_count = len(server.get("tools", []))

                table.add_row(
                    server.get("name", "Unknown"),
                    status_text,
                    str(tools_count)
                )

            self.console.print(table)
        else:
            print("\nMCP Servers:")
            print("=" * 40)
            for server in servers:
                status = server.get("status", "Unknown")
                tools_count = len(server.get("tools", []))
                print(f"• {server.get('name', 'Unknown')}: {status} ({tools_count} tools)")

    def show_saved_conversations(self, conversations: List[str]):
        """Show saved conversation tags."""
        if self.use_rich:
            if conversations:
                table = Table(title="Saved Conversations")
                table.add_column("Tag", style="cyan")

                for tag in conversations:
                    table.add_row(tag)

                self.console.print(table)
            else:
                self.console.print("[yellow]No saved conversations[/yellow]")
        else:
            print("\nSaved Conversations:")
            if conversations:
                for tag in conversations:
                    print(f"• {tag}")
            else:
                print("No saved conversations")

    def show_checkpoints(self, checkpoints: List[Dict[str, Any]]):
        """Show available checkpoints."""
        if self.use_rich:
            if checkpoints:
                table = Table(title="Available Checkpoints")
                table.add_column("ID", style="cyan")
                table.add_column("Tool", style="white")
                table.add_column("Created", style="dim")

                for checkpoint in checkpoints:
                    table.add_row(
                        checkpoint.get("id", ""),
                        checkpoint.get("tool_name", ""),
                        format_timestamp(checkpoint.get("timestamp"))
                    )

                self.console.print(table)
            else:
                self.console.print("[yellow]No checkpoints available[/yellow]")
        else:
            print("\nAvailable Checkpoints:")
            if checkpoints:
                for checkpoint in checkpoints:
                    print(f"• {checkpoint.get('id', '')}: {checkpoint.get('tool_name', '')} "
                          f"({format_timestamp(checkpoint.get('timestamp'))})")
            else:
                print("No checkpoints available")

    def show_theme_selector(self, themes: List[str]):
        """Show theme selection dialog."""
        if self.use_rich:
            self.console.print("[yellow]Available Themes:[/yellow]")
            for i, theme in enumerate(themes, 1):
                current = " (current)" if theme == self.theme_name else ""
                self.console.print(f"{i}. {theme}{current}")
        else:
            print("Available Themes:")
            for i, theme in enumerate(themes, 1):
                current = " (current)" if theme == self.theme_name else ""
                print(f"{i}. {theme}{current}")

    def show_editor_selector(self, editors: List[str]):
        """Show editor selection dialog."""
        if self.use_rich:
            self.console.print("[yellow]Available Editors:[/yellow]")
            for i, editor in enumerate(editors, 1):
                self.console.print(f"{i}. {editor}")
        else:
            print("Available Editors:")
            for i, editor in enumerate(editors, 1):
                print(f"{i}. {editor}")

    def show_auth_info(self, auth_config: Dict[str, Any]):
        """Show authentication information."""
        auth_type = auth_config.get("type", "local")

        if self.use_rich:
            info_text = f"Authentication Type: [cyan]{auth_type}[/cyan]"
            if auth_type == "api_key":
                has_key = bool(auth_config.get("apiKey"))
                info_text += f"\nAPI Key: [{'green' if has_key else 'red'}]{'Set' if has_key else 'Not Set'}[/{'green' if has_key else 'red'}]"

            panel = Panel(info_text, title="[blue]Authentication[/blue]", border_style="blue")
            self.console.print(panel)
        else:
            print(f"\nAuthentication Type: {auth_type}")
            if auth_type == "api_key":
                has_key = bool(auth_config.get("apiKey"))
                print(f"API Key: {'Set' if has_key else 'Not Set'}")

    def show_extensions(self, extensions: List[Dict[str, Any]]):
        """Show extension information."""
        if self.use_rich:
            if extensions:
                table = Table(title="Extensions")
                table.add_column("Name", style="cyan")
                table.add_column("Version", style="white")
                table.add_column("Status", style="white")
                table.add_column("Description", style="dim")

                for ext in extensions:
                    status_color = "green" if ext.get("enabled") else "red"
                    status_text = f"[{status_color}]{'Enabled' if ext.get('enabled') else 'Disabled'}[/{status_color}]"

                    table.add_row(
                        ext.get("name", ""),
                        ext.get("version", ""),
                        status_text,
                        ext.get("description", "")[:50] + ("..." if len(ext.get("description", "")) > 50 else "")
                    )

                self.console.print(table)
            else:
                self.console.print("[yellow]No extensions loaded[/yellow]")
        else:
            print("\nExtensions:")
            if extensions:
                for ext in extensions:
                    status = "Enabled" if ext.get("enabled") else "Disabled"
                    print(f"• {ext.get('name', '')} ({ext.get('version', '')}) - {status}")
                    if ext.get("description"):
                        print(f"  {ext.get('description', '')}")
            else:
                print("No extensions loaded")

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text

        color_codes = {
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bold": "\033[1m",
            "bold_red": "\033[1;31m",
            "bold_green": "\033[1;32m",
            "bold_yellow": "\033[1;33m",
            "bold_blue": "\033[1;34m",
            "reset": "\033[0m"
        }

        color_code = color_codes.get(color, "")
        reset_code = color_codes["reset"]

        return f"{color_code}{text}{reset_code}"

    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text looks like markdown."""
        markdown_indicators = [
            "# ", "## ", "### ", "#### ", "##### ", "###### ",  # Headers
            "- ", "* ", "+ ",  # Lists
            "```",  # Code blocks
            "`",    # Inline code
            "**", "__",  # Bold
            "*", "_",    # Italic
            "[", "](",   # Links
        ]

        return any(indicator in text for indicator in markdown_indicators)

    def _show_usage_info(self, usage: Dict[str, Any]):
        """Show token usage information."""
        if not usage:
            return

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        if self.use_rich:
            usage_text = f"Tokens: [dim]{input_tokens} in, {output_tokens} out, {total_tokens} total[/dim]"
            self.console.print(usage_text)
        else:
            print(f"Tokens: {input_tokens} in, {output_tokens} out, {total_tokens} total")

    def _show_structured_result(self, result: Dict[str, Any]):
        """Show structured tool result."""
        if self.use_rich:
            # Format as JSON with syntax highlighting
            json_str = json.dumps(result, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai")
            self.console.print(syntax)
        else:
            print(json.dumps(result, indent=2))

    def _format_final_stats(self, stats: Dict[str, Any]) -> str:
        """Format final statistics for rich display."""
        lines = []
        lines.append(f"Session Duration: [cyan]{format_duration(stats.get('session_duration', 0))}[/cyan]")
        lines.append(f"Total Prompts: [cyan]{stats.get('total_prompts', 0)}[/cyan]")
        lines.append(f"Total Tool Calls: [cyan]{stats.get('total_tool_calls', 0)}[/cyan]")

        if "token_usage" in stats:
            usage = stats["token_usage"]
            lines.append(f"Total Tokens: [cyan]{usage.get('total_tokens', 0)}[/cyan]")

        return "\n".join(lines)

    def _format_final_stats_plain(self, stats: Dict[str, Any]) -> str:
        """Format final statistics for plain text display."""
        lines = []
        lines.append(f"Session Duration: {format_duration(stats.get('session_duration', 0))}")
        lines.append(f"Total Prompts: {stats.get('total_prompts', 0)}")
        lines.append(f"Total Tool Calls: {stats.get('total_tool_calls', 0)}")

        if "token_usage" in stats:
            usage = stats["token_usage"]
            lines.append(f"Total Tokens: {usage.get('total_tokens', 0)}")

        return "\n".join(lines)