"""
Shell utilities for Gemini CLI
Provides shell mode and command execution utilities
"""

import os
import shlex
import subprocess
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging


class ShellMode:
    """Manages shell mode state for the CLI."""

    def __init__(self):
        self.active = False
        self.history: List[str] = []
        self.working_directory = Path.cwd()
        self.environment = dict(os.environ)
        self.logger = logging.getLogger(f"{__name__}.ShellMode")

    def toggle(self) -> bool:
        """Toggle shell mode on/off."""
        self.active = not self.active
        return self.active

    def activate(self):
        """Activate shell mode."""
        self.active = True

    def deactivate(self):
        """Deactivate shell mode."""
        self.active = False

    def add_to_history(self, command: str):
        """Add command to shell history."""
        if command.strip() and (not self.history or self.history[-1] != command):
            self.history.append(command)

        # Keep history size reasonable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def get_recent_commands(self, count: int = 10) -> List[str]:
        """Get recent commands from history."""
        return self.history[-count:] if self.history else []

    def set_working_directory(self, path: Path):
        """Set the working directory for shell commands."""
        if path.exists() and path.is_dir():
            self.working_directory = path.resolve()
            os.chdir(self.working_directory)

    def get_working_directory(self) -> Path:
        """Get current working directory."""
        return self.working_directory

    def set_environment_variable(self, key: str, value: str):
        """Set environment variable."""
        self.environment[key] = value
        os.environ[key] = value

    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return self.environment.get(key, default)


class ShellCommandParser:
    """Parses and validates shell commands."""

    def __init__(self):
        self.dangerous_commands = {
            'rm', 'rmdir', 'del', 'deltree', 'format', 'fdisk',
            'mkfs', 'dd', 'shutdown', 'reboot', 'halt', 'poweroff',
            'kill', 'killall', 'pkill', 'taskkill'
        }

        self.builtin_commands = {
            'cd', 'pwd', 'ls', 'dir', 'echo', 'set', 'export',
            'history', 'exit', 'clear', 'cls'
        }

    def parse_command(self, command_line: str) -> Dict[str, Any]:
        """Parse a command line into components."""
        try:
            # Handle empty or whitespace-only commands
            if not command_line.strip():
                return {
                    'valid': False,
                    'error': 'Empty command',
                    'command': '',
                    'args': [],
                    'full_line': command_line
                }

            # Parse using shlex for proper handling of quotes and escapes
            parts = shlex.split(command_line)

            if not parts:
                return {
                    'valid': False,
                    'error': 'No command specified',
                    'command': '',
                    'args': [],
                    'full_line': command_line
                }

            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            return {
                'valid': True,
                'command': command,
                'args': args,
                'full_line': command_line,
                'is_builtin': command in self.builtin_commands,
                'is_dangerous': command in self.dangerous_commands,
                'requires_confirmation': command in self.dangerous_commands
            }

        except ValueError as e:
            return {
                'valid': False,
                'error': f'Parse error: {e}',
                'command': '',
                'args': [],
                'full_line': command_line
            }

    def is_safe_command(self, command: str) -> bool:
        """Check if a command is considered safe."""
        return command not in self.dangerous_commands

    def needs_confirmation(self, command: str) -> bool:
        """Check if a command needs user confirmation."""
        return command in self.dangerous_commands


class ShellExecutor:
    """Executes shell commands with proper error handling."""

    def __init__(self, shell_mode: ShellMode):
        self.shell_mode = shell_mode
        self.parser = ShellCommandParser()
        self.logger = logging.getLogger(f"{__name__}.ShellExecutor")

    async def execute_command(self, command_line: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command and return results."""
        # Parse the command
        parsed = self.parser.parse_command(command_line)

        if not parsed['valid']:
            return {
                'success': False,
                'error': parsed['error'],
                'command': command_line,
                'exit_code': -1,
                'stdout': '',
                'stderr': parsed['error'],
                'execution_time': 0.0
            }

        # Add to history
        self.shell_mode.add_to_history(command_line)

        # Handle builtin commands
        if parsed['is_builtin']:
            return await self._execute_builtin(parsed)

        # Execute external command
        return await self._execute_external(parsed, timeout)

    async def _execute_builtin(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Execute builtin shell commands."""
        command = parsed['command']
        args = parsed['args']

        try:
            if command == 'cd':
                return await self._builtin_cd(args)
            elif command == 'pwd':
                return await self._builtin_pwd()
            elif command in ('ls', 'dir'):
                return await self._builtin_ls(args)
            elif command == 'echo':
                return await self._builtin_echo(args)
            elif command in ('set', 'export'):
                return await self._builtin_set(args)
            elif command == 'history':
                return await self._builtin_history(args)
            elif command in ('clear', 'cls'):
                return await self._builtin_clear()
            else:
                return {
                    'success': False,
                    'error': f'Builtin command {command} not implemented',
                    'command': parsed['full_line'],
                    'exit_code': -1,
                    'stdout': '',
                    'stderr': f'Command not implemented: {command}',
                    'execution_time': 0.0
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': parsed['full_line'],
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0.0
            }

    async def _execute_external(self, parsed: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Execute external shell commands."""
        import time

        start_time = time.time()

        try:
            # Create the subprocess
            process = await asyncio.create_subprocess_exec(
                parsed['command'],
                *parsed['args'],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.shell_mode.working_directory,
                env=self.shell_mode.environment
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution_time = time.time() - start_time

                return {
                    'success': False,
                    'error': f'Command timed out after {timeout} seconds',
                    'command': parsed['full_line'],
                    'exit_code': -1,
                    'stdout': '',
                    'stderr': f'Command timed out after {timeout} seconds',
                    'execution_time': execution_time
                }

            execution_time = time.time() - start_time

            return {
                'success': process.returncode == 0,
                'command': parsed['full_line'],
                'exit_code': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'execution_time': execution_time
            }

        except FileNotFoundError:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': f'Command not found: {parsed["command"]}',
                'command': parsed['full_line'],
                'exit_code': 127,
                'stdout': '',
                'stderr': f'Command not found: {parsed["command"]}',
                'execution_time': execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'command': parsed['full_line'],
                'exit_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': execution_time
            }

    async def _builtin_cd(self, args: List[str]) -> Dict[str, Any]:
        """Implement cd command."""
        if not args:
            # cd with no arguments goes to home directory
            target = Path.home()
        elif args[0] == '-':
            # cd - goes to previous directory (simplified)
            target = Path.home()  # Could implement previous directory tracking
        else:
            target = Path(args[0])

        try:
            if not target.exists():
                return {
                    'success': False,
                    'error': f'Directory not found: {target}',
                    'command': f'cd {args[0] if args else ""}',
                    'exit_code': 1,
                    'stdout': '',
                    'stderr': f'cd: {target}: No such file or directory',
                    'execution_time': 0.0
                }

            if not target.is_dir():
                return {
                    'success': False,
                    'error': f'Not a directory: {target}',
                    'command': f'cd {args[0] if args else ""}',
                    'exit_code': 1,
                    'stdout': '',
                    'stderr': f'cd: {target}: Not a directory',
                    'execution_time': 0.0
                }

            self.shell_mode.set_working_directory(target)

            return {
                'success': True,
                'command': f'cd {args[0] if args else ""}',
                'exit_code': 0,
                'stdout': '',
                'stderr': '',
                'execution_time': 0.0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': f'cd {args[0] if args else ""}',
                'exit_code': 1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0.0
            }

    async def _builtin_pwd(self) -> Dict[str, Any]:
        """Implement pwd command."""
        cwd = str(self.shell_mode.get_working_directory())

        return {
            'success': True,
            'command': 'pwd',
            'exit_code': 0,
            'stdout': cwd + '\n',
            'stderr': '',
            'execution_time': 0.0
        }

    async def _builtin_ls(self, args: List[str]) -> Dict[str, Any]:
        """Implement ls/dir command."""
        try:
            target = Path(args[0]) if args else self.shell_mode.working_directory

            if not target.exists():
                return {
                    'success': False,
                    'error': f'Path not found: {target}',
                    'command': f'ls {args[0] if args else ""}',
                    'exit_code': 1,
                    'stdout': '',
                    'stderr': f'ls: {target}: No such file or directory',
                    'execution_time': 0.0
                }

            if target.is_file():
                output = str(target.name) + '\n'
            else:
                entries = []
                for item in sorted(target.iterdir()):
                    if item.is_dir():
                        entries.append(f'{item.name}/')
                    else:
                        entries.append(item.name)
                output = '\n'.join(entries) + '\n' if entries else ''

            return {
                'success': True,
                'command': f'ls {args[0] if args else ""}',
                'exit_code': 0,
                'stdout': output,
                'stderr': '',
                'execution_time': 0.0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': f'ls {args[0] if args else ""}',
                'exit_code': 1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0.0
            }

    async def _builtin_echo(self, args: List[str]) -> Dict[str, Any]:
        """Implement echo command."""
        output = ' '.join(args) + '\n'

        return {
            'success': True,
            'command': f'echo {" ".join(args)}',
            'exit_code': 0,
            'stdout': output,
            'stderr': '',
            'execution_time': 0.0
        }

    async def _builtin_set(self, args: List[str]) -> Dict[str, Any]:
        """Implement set/export command."""
        if not args:
            # Show all environment variables
            output = '\n'.join(f'{k}={v}' for k, v in self.shell_mode.environment.items()) + '\n'
        else:
            # Set environment variable
            for arg in args:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    self.shell_mode.set_environment_variable(key, value)
                else:
                    # Show specific variable
                    value = self.shell_mode.get_environment_variable(arg)
                    if value is not None:
                        output = f'{arg}={value}\n'
                    else:
                        return {
                            'success': False,
                            'error': f'Variable not set: {arg}',
                            'command': f'set {" ".join(args)}',
                            'exit_code': 1,
                            'stdout': '',
                            'stderr': f'Variable not set: {arg}',
                            'execution_time': 0.0
                        }

            output = ''

        return {
            'success': True,
            'command': f'set {" ".join(args)}',
            'exit_code': 0,
            'stdout': output,
            'stderr': '',
            'execution_time': 0.0
        }

    async def _builtin_history(self, args: List[str]) -> Dict[str, Any]:
        """Implement history command."""
        count = 10
        if args and args[0].isdigit():
            count = int(args[0])

        history = self.shell_mode.get_recent_commands(count)
        output = '\n'.join(f'{i+1:4d}  {cmd}' for i, cmd in enumerate(history)) + '\n'

        return {
            'success': True,
            'command': f'history {" ".join(args)}',
            'exit_code': 0,
            'stdout': output,
            'stderr': '',
            'execution_time': 0.0
        }

    async def _builtin_clear(self) -> Dict[str, Any]:
        """Implement clear/cls command."""
        # Return escape sequence to clear screen
        return {
            'success': True,
            'command': 'clear',
            'exit_code': 0,
            'stdout': '\033[2J\033[H',  # Clear screen and move cursor to top
            'stderr': '',
            'execution_time': 0.0
        }