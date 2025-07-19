"""
Sandbox manager for Gemini CLI core
Provides high-level sandbox management and configuration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .exceptions import SandboxError
from ..utils.sandbox import SandboxExecutor


class SandboxManager:
    """High-level sandbox management for the CLI."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sandbox_config = config.get("sandbox", {})
        self.logger = logging.getLogger(__name__)

        # Initialize the sandbox executor
        self.executor = SandboxExecutor(config)

        # Manager state
        self.initialized = False
        self.active_executions: List[str] = []

    async def initialize(self):
        """Initialize the sandbox manager."""
        if self.initialized:
            return

        try:
            # Check if sandbox is enabled
            if not self.sandbox_config.get("enabled", False):
                self.logger.info("Sandbox is disabled")
                self.initialized = True
                return

            # Validate sandbox configuration
            await self._validate_configuration()

            # Check sandbox availability
            await self._check_sandbox_availability()

            # Initialize base image if needed
            await self._ensure_base_image()

            self.initialized = True
            self.logger.info("Sandbox manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize sandbox manager: {e}")
            raise SandboxError(f"Sandbox initialization failed: {e}")

    async def execute_command(self, command: str, working_dir: Optional[Path] = None, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute a command in the sandbox."""
        if not self.initialized:
            await self.initialize()

        if working_dir is None:
            working_dir = Path.cwd()

        try:
            # Track active execution
            execution_id = f"cmd_{len(self.active_executions)}"
            self.active_executions.append(execution_id)

            # Execute the command
            result = await self.executor.execute(command, working_dir, environment)

            # Add sandbox manager metadata
            result["execution_id"] = execution_id
            result["managed_by"] = "SandboxManager"

            return result

        except Exception as e:
            self.logger.error(f"Sandbox command execution failed: {e}")
            raise SandboxError(f"Command execution failed: {e}")
        finally:
            # Remove from active executions
            if execution_id in self.active_executions:
                self.active_executions.remove(execution_id)

    async def execute_code(self, code: str, language: str, working_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute code in the sandbox."""
        if not self.initialized:
            await self.initialize()

        try:
            # Track active execution
            execution_id = f"code_{len(self.active_executions)}"
            self.active_executions.append(execution_id)

            # Execute the code
            result = await self.executor.execute_code(code, language, working_dir)

            # Add sandbox manager metadata
            result["execution_id"] = execution_id
            result["language"] = language
            result["managed_by"] = "SandboxManager"

            return result

        except Exception as e:
            self.logger.error(f"Sandbox code execution failed: {e}")
            raise SandboxError(f"Code execution failed: {e}")
        finally:
            # Remove from active executions
            if execution_id in self.active_executions:
                self.active_executions.remove(execution_id)

    async def cleanup(self):
        """Clean up sandbox resources."""
        try:
            if self.executor:
                await self.executor.cleanup()

            self.active_executions.clear()
            self.logger.info("Sandbox manager cleaned up")

        except Exception as e:
            self.logger.error(f"Error during sandbox cleanup: {e}")

    def is_enabled(self) -> bool:
        """Check if sandbox is enabled."""
        return self.sandbox_config.get("enabled", False)

    def get_status(self) -> Dict[str, Any]:
        """Get sandbox manager status."""
        return {
            "enabled": self.is_enabled(),
            "initialized": self.initialized,
            "active_executions": len(self.active_executions),
            "sandbox_type": self.sandbox_config.get("type", "docker"),
            "image": self.sandbox_config.get("image", "gemini-cli-sandbox"),
            "executor_info": self.executor.get_sandbox_info() if self.executor else None
        }

    async def _validate_configuration(self):
        """Validate sandbox configuration."""
        required_fields = ["type", "image"]

        for field in required_fields:
            if field not in self.sandbox_config:
                raise SandboxError(f"Missing required sandbox configuration: {field}")

        sandbox_type = self.sandbox_config["type"]
        if sandbox_type not in ["docker", "podman", "sandbox-exec"]:
            raise SandboxError(f"Unsupported sandbox type: {sandbox_type}")

        # Validate memory and CPU limits
        memory_limit = self.sandbox_config.get("memoryLimit", "512m")
        if not isinstance(memory_limit, str) or not memory_limit:
            raise SandboxError("Invalid memory limit configuration")

        cpu_limit = self.sandbox_config.get("cpuLimit", "0.5")
        try:
            float(cpu_limit)
        except (ValueError, TypeError):
            raise SandboxError("Invalid CPU limit configuration")

    async def _check_sandbox_availability(self):
        """Check if the sandbox system is available."""
        sandbox_type = self.sandbox_config["type"]

        if sandbox_type == "docker":
            if not await self.executor._check_docker():
                raise SandboxError("Docker is not available or not running")

        elif sandbox_type == "podman":
            if not await self.executor._check_podman():
                raise SandboxError("Podman is not available or not running")

        elif sandbox_type == "sandbox-exec":
            # Check if we're on macOS
            import platform
            if platform.system() != "Darwin":
                raise SandboxError("sandbox-exec is only available on macOS")

    async def _ensure_base_image(self):
        """Ensure the base sandbox image exists."""
        sandbox_type = self.sandbox_config["type"]
        image = self.sandbox_config["image"]

        if sandbox_type in ["docker", "podman"]:
            # Check if image exists
            try:
                check_cmd = [sandbox_type, "image", "inspect", image]
                process = await asyncio.create_subprocess_exec(
                    *check_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await process.communicate()

                if process.returncode != 0:
                    # Image doesn't exist, try to pull it
                    self.logger.info(f"Pulling sandbox image: {image}")
                    await self._pull_image(image, sandbox_type)
                else:
                    self.logger.info(f"Sandbox image {image} is available")

            except Exception as e:
                self.logger.warning(f"Could not verify sandbox image: {e}")
                # Continue anyway - image might be available

    async def _pull_image(self, image: str, sandbox_type: str):
        """Pull a sandbox image."""
        try:
            pull_cmd = [sandbox_type, "pull", image]
            process = await asyncio.create_subprocess_exec(
                *pull_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace')
                raise SandboxError(f"Failed to pull image {image}: {error_msg}")

            self.logger.info(f"Successfully pulled sandbox image: {image}")

        except Exception as e:
            # If we can't pull the image, try to build a basic one
            self.logger.warning(f"Could not pull image {image}: {e}")
            await self._create_basic_image(image, sandbox_type)

    async def _create_basic_image(self, image: str, sandbox_type: str):
        """Create a basic sandbox image."""
        try:
            dockerfile_content = """
FROM ubuntu:22.04

# Install basic tools
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    nodejs \\
    npm \\
    curl \\
    wget \\
    git \\
    vim \\
    nano \\
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]
"""

            success = await self.executor.build_sandbox_image(dockerfile_content, image)

            if success:
                self.logger.info(f"Created basic sandbox image: {image}")
            else:
                self.logger.warning(f"Could not create sandbox image: {image}")

        except Exception as e:
            self.logger.warning(f"Failed to create basic sandbox image: {e}")

    async def test_sandbox(self) -> Dict[str, Any]:
        """Test the sandbox functionality."""
        if not self.initialized:
            await self.initialize()

        tests = []

        # Test basic command execution
        try:
            result = await self.execute_command("echo 'Hello, Sandbox!'")
            tests.append({
                "test": "basic_command",
                "success": result["exit_code"] == 0,
                "output": result["stdout"].strip(),
                "error": result.get("stderr", "")
            })
        except Exception as e:
            tests.append({
                "test": "basic_command",
                "success": False,
                "error": str(e)
            })

        # Test Python code execution
        try:
            python_code = "print('Python works in sandbox')"
            result = await self.execute_code(python_code, "python")
            tests.append({
                "test": "python_code",
                "success": result["exit_code"] == 0,
                "output": result["stdout"].strip(),
                "error": result.get("stderr", "")
            })
        except Exception as e:
            tests.append({
                "test": "python_code",
                "success": False,
                "error": str(e)
            })

        # Test file operations
        try:
            result = await self.execute_command("touch test_file.txt && ls test_file.txt")
            tests.append({
                "test": "file_operations",
                "success": result["exit_code"] == 0,
                "output": result["stdout"].strip(),
                "error": result.get("stderr", "")
            })
        except Exception as e:
            tests.append({
                "test": "file_operations",
                "success": False,
                "error": str(e)
            })

        # Test network restrictions (if enabled)
        if not self.sandbox_config.get("allowNetwork", False):
            try:
                result = await self.execute_command("curl -m 5 https://httpbin.org/get")
                # This should fail if network is properly restricted
                tests.append({
                    "test": "network_restriction",
                    "success": result["exit_code"] != 0,  # Should fail
                    "output": result["stdout"].strip(),
                    "error": result.get("stderr", "")
                })
            except Exception as e:
                tests.append({
                    "test": "network_restriction",
                    "success": True,  # Exception is expected with network restrictions
                    "error": str(e)
                })

        overall_success = all(test["success"] for test in tests)

        return {
            "overall_success": overall_success,
            "tests": tests,
            "sandbox_info": self.get_status()
        }

    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage information for sandbox."""
        # This would need to be implemented based on the sandbox type
        # For now, return basic information
        return {
            "active_executions": len(self.active_executions),
            "sandbox_type": self.sandbox_config.get("type"),
            "memory_limit": self.sandbox_config.get("memoryLimit"),
            "cpu_limit": self.sandbox_config.get("cpuLimit")
        }

    async def restart_sandbox(self):
        """Restart the sandbox system."""
        try:
            self.logger.info("Restarting sandbox...")

            # Cleanup current resources
            await self.cleanup()

            # Reset state
            self.initialized = False
            self.active_executions.clear()

            # Reinitialize
            await self.initialize()

            self.logger.info("Sandbox restarted successfully")

        except Exception as e:
            self.logger.error(f"Failed to restart sandbox: {e}")
            raise SandboxError(f"Sandbox restart failed: {e}")

    def get_configuration(self) -> Dict[str, Any]:
        """Get the current sandbox configuration."""
        return self.sandbox_config.copy()

    async def update_configuration(self, new_config: Dict[str, Any]):
        """Update sandbox configuration."""
        try:
            # Validate new configuration
            old_config = self.sandbox_config.copy()
            self.sandbox_config.update(new_config)

            try:
                await self._validate_configuration()
            except SandboxError:
                # Restore old configuration if validation fails
                self.sandbox_config = old_config
                raise

            # If configuration changed significantly, restart
            if (old_config.get("type") != new_config.get("type") or
                old_config.get("image") != new_config.get("image")):
                await self.restart_sandbox()

            self.logger.info("Sandbox configuration updated")

        except Exception as e:
            self.logger.error(f"Failed to update sandbox configuration: {e}")
            raise SandboxError(f"Configuration update failed: {e}")