"""
Main entry point for Gemini CLI
Handles command line argument parsing and initialization
"""

import asyncio
import argparse
import sys
import os
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any

from .core.config import ConfigManager
from .core.cli import GeminiCLI
from .core.exceptions import GeminiCLIError
from .utils.version import get_version


def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemini-cli",
        description="Interactive AI assistant powered by Gemini OpenSource",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gemini-cli                           # Start interactive mode
  gemini-cli "Hello, world!"          # Single prompt mode
  gemini-cli --config custom.json     # Use custom config
  gemini-cli --model gemini-large     # Use specific model
  gemini-cli --version                # Show version

For more information, visit: https://github.com/kyegomez/Gemini
        """
    )

    # Positional arguments
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single prompt to execute (enters non-interactive mode)"
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to use (overrides config)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0.0 to 2.0)"
    )

    parser.add_argument(
        "--theme",
        type=str,
        choices=["Default", "Dark", "Light", "Monokai", "GitHub"],
        help="Color theme for the interface"
    )

    # Tool configuration
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="Automatically accept tool executions"
    )

    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable sandbox mode for tool execution"
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory loading"
    )

    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable all tools"
    )

    parser.add_argument(
        "--core-tools",
        type=str,
        nargs="+",
        help="Specify which tools to enable"
    )

    parser.add_argument(
        "--exclude-tools",
        type=str,
        nargs="+",
        help="Specify which tools to exclude"
    )

    # File operations
    parser.add_argument(
        "--include-files",
        type=str,
        nargs="+",
        help="Files to include in context"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for single prompt mode"
    )

    # Debugging and development
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    # Version and help
    parser.add_argument(
        "--version",
        action="version",
        version=f"Gemini CLI {get_version()}"
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit"
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )

    parser.add_argument(
        "--list-themes",
        action="store_true",
        help="List available themes and exit"
    )

    return parser


def setup_environment(args: argparse.Namespace):
    """Setup environment variables based on arguments."""
    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    if args.debug:
        os.environ["GEMINI_DEBUG"] = "true"

    if args.verbose:
        os.environ["GEMINI_VERBOSE"] = "true"

    if args.log_level:
        os.environ["GEMINI_LOG_LEVEL"] = args.log_level

    if args.auto_accept:
        os.environ["GEMINI_AUTO_ACCEPT"] = "true"

    if args.sandbox:
        os.environ["GEMINI_SANDBOX"] = "true"


def create_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration overrides from command line arguments."""
    overrides = {}

    if args.model:
        overrides["model"] = args.model

    if args.max_tokens:
        overrides["maxTokens"] = args.max_tokens

    if args.temperature:
        overrides["temperature"] = args.temperature

    if args.theme:
        overrides["theme"] = args.theme

    if args.auto_accept:
        overrides["autoAccept"] = True

    if args.sandbox:
        overrides["sandbox"] = {"enabled": True}

    if args.no_memory:
        overrides["memory"] = {"enabled": False}

    if args.no_tools:
        overrides["coreTools"] = []
    elif args.core_tools:
        overrides["coreTools"] = args.core_tools

    if args.exclude_tools:
        overrides["excludeTools"] = args.exclude_tools

    if args.debug:
        overrides["debug"] = True

    if args.verbose:
        overrides["verbose"] = True

    if args.log_level:
        overrides["logLevel"] = args.log_level

    return overrides


async def run_health_check(cli: GeminiCLI) -> int:
    """Run health check and return exit code."""
    print("Running Gemini CLI health check...")

    try:
        # Check model health
        model_healthy = await cli.model_interface.health_check()
        print(f"Model interface: {'✓ OK' if model_healthy else '✗ FAIL'}")

        # Check tool registry
        tools = cli.tool_registry.list_tools()
        print(f"Tools loaded: {len(tools)} tools")

        # Check configuration
        config_issues = cli.config_manager.validate_config()
        if config_issues:
            print(f"Configuration issues: {len(config_issues)}")
            for issue in config_issues:
                print(f"  - {issue}")
        else:
            print("Configuration: ✓ OK")

        # Check memory system
        if cli.memory_manager.enabled:
            memory_stats = cli.memory_manager.get_memory_stats()
            print(f"Memory system: ✓ OK ({memory_stats['loaded_files']} files loaded)")
        else:
            print("Memory system: Disabled")

        overall_health = model_healthy and not config_issues
        print(f"\nOverall health: {'✓ HEALTHY' if overall_health else '✗ UNHEALTHY'}")

        return 0 if overall_health else 1

    except Exception as e:
        print(f"Health check failed: {e}")
        return 1


async def list_tools(cli: GeminiCLI) -> int:
    """List available tools and return exit code."""
    try:
        tools = cli.tool_registry.list_tools(include_descriptions=True)

        if not tools:
            print("No tools available")
            return 1

        print(f"Available Tools ({len(tools)}):")
        print("=" * 50)

        for tool in tools:
            print(f"\n{tool['name']}")
            print(f"  Description: {tool['description']}")

            if 'schema' in tool and 'parameters' in tool['schema']:
                params = tool['schema']['parameters']
                if 'properties' in params:
                    print("  Parameters:")
                    for param_name, param_info in params['properties'].items():
                        required = param_name in params.get('required', [])
                        req_marker = " (required)" if required else ""
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', 'No description')
                        print(f"    - {param_name} ({param_type}){req_marker}: {param_desc}")

        return 0

    except Exception as e:
        print(f"Failed to list tools: {e}")
        return 1


def list_themes() -> int:
    """List available themes and return exit code."""
    themes = ["Default", "Dark", "Light", "Monokai", "GitHub"]

    print("Available Themes:")
    print("=" * 20)
    for theme in themes:
        print(f"  - {theme}")

    return 0


async def run_single_prompt(cli: GeminiCLI, prompt: str, include_files: Optional[List[str]] = None, output_file: Optional[str] = None) -> int:
    """Run a single prompt and exit."""
    try:
        # Include files if specified
        context_files = []
        if include_files:
            for file_pattern in include_files:
                files = cli.file_discovery.discover_files(file_pattern)
                context_files.extend(files)

        # Execute prompt
        response = await cli.execute_prompt(prompt, context_files=context_files)

        # Output result
        result_text = response.get("content", "")

        if output_file:
            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)

            print(f"Output written to: {output_file}")
        else:
            # Print to stdout
            print(result_text)

        return 0

    except Exception as e:
        print(f"Error executing prompt: {e}", file=sys.stderr)
        return 1


async def setup_signal_handlers(cli: GeminiCLI):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print("\nShutting down...")
        cli.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main_async(args: Optional[List[str]] = None) -> int:
    """Main async function."""
    # Parse arguments
    parser = create_arg_parser()
    parsed_args = parser.parse_args(args)

    # Setup environment
    setup_environment(parsed_args)

    try:
        # Load configuration
        config_manager = ConfigManager(parsed_args.config)

        # Apply command line overrides
        overrides = create_config_overrides(parsed_args)
        if overrides:
            config_manager.apply_overrides(overrides)

        # Initialize CLI
        cli = GeminiCLI(config_manager)
        await cli.initialize()

        # Setup signal handlers
        await setup_signal_handlers(cli)

        # Handle special modes
        if parsed_args.health_check:
            return await run_health_check(cli)

        if parsed_args.list_tools:
            return await list_tools(cli)

        if parsed_args.list_themes:
            return list_themes()

        # Handle single prompt mode
        if parsed_args.prompt:
            return await run_single_prompt(
                cli,
                parsed_args.prompt,
                parsed_args.include_files,
                parsed_args.output
            )

        # Run interactive mode
        print(f"Starting Gemini CLI {get_version()}")
        return await cli.run_interactive()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except GeminiCLIError as e:
        print(f"CLI Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if parsed_args.debug:
            import traceback
            traceback.print_exc()
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())