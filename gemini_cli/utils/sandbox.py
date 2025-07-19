"""
Sandbox execution utilities for Gemini CLI
Provides isolated execution environments for shell commands and code
"""

import asyncio
import subprocess
import tempfile
import shutil
import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from ..core.exceptions import SandboxError


class SandboxExecutor:
    """Executes commands and code in isolated sandbox environments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sandbox_config = config.get("sandbox", {})
        self.logger = logging.getLogger(__name__)

        # Sandbox configuration
        self.enabled = self.sandbox_config.get("enabled", False)
        self.sandbox_type = self.sandbox_config.get("type", "docker")
        self.image = self.sandbox_config.get("image", "gemini-cli-sandbox")
        self.timeout = self.sandbox_config.get("timeout", 30)
        self.memory_limit = self.sandbox_config.get("memoryLimit", "512m")
        self.cpu_limit = self.sandbox_config.get("cpuLimit", "0.5")

        # Working directory management
        self.temp_dirs: List[Path] = []
        self.containers: List[str] = []

        # Security settings
        self.allowed_network = self.sandbox_config.get("allowNetwork", False)
        self.read_only_filesystem = self.sandbox_config.get("readOnlyFilesystem", True)
        self.drop_capabilities = self.sandbox_config.get("dropCapabilities", True)

    async def execute(self, command: str, working_dir: Path, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute a command in the sandbox."""
        if not self.enabled:
            return await self._execute_native(command, working_dir, environment)

        if self.sandbox_type == "docker":
            return await self._execute_docker(command, working_dir, environment)
        elif self.sandbox_type == "podman":
            return await self._execute_podman(command, working_dir, environment)
        elif self.sandbox_type == "sandbox-exec":
            return await self._execute_sandbox_exec(command, working_dir, environment)
        else:
            raise SandboxError(f"Unsupported sandbox type: {self.sandbox_type}")

    async def execute_code(self, code: str, language: str, working_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute code in the sandbox."""
        if working_dir is None:
            working_dir = Path.cwd()

        # Create temporary file for code
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)

        try:
            if language.lower() == "python":
                code_file = temp_dir / "script.py"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                command = f"python {code_file}"

            elif language.lower() in ("javascript", "js", "node"):
                code_file = temp_dir / "script.js"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                command = f"node {code_file}"

            elif language.lower() == "bash":
                code_file = temp_dir / "script.sh"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                os.chmod(code_file, 0o755)
                command = f"bash {code_file}"

            else:
                raise SandboxError(f"Unsupported language: {language}")

            return await self.execute(command, working_dir)

        finally:
            # Cleanup is handled by cleanup() method
            pass

    async def _execute_native(self, command: str, working_dir: Path, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute command natively without sandbox."""
        start_time = time.time()

        try:
            # Prepare environment
            env = dict(os.environ)
            if environment:
                env.update(environment)

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise SandboxError(f"Command timed out after {self.timeout} seconds")

            execution_time = time.time() - start_time

            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "execution_time": execution_time,
                "sandbox_type": "native"
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": execution_time,
                "error": str(e),
                "sandbox_type": "native"
            }

    async def _execute_docker(self, command: str, working_dir: Path, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute command in Docker container."""
        start_time = time.time()

        try:
            # Check if Docker is available
            if not await self._check_docker():
                raise SandboxError("Docker is not available")

            # Prepare Docker command
            docker_args = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "--interactive",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                f"--workdir=/workspace",
                f"--volume={working_dir}:/workspace"
            ]

            # Security options
            if self.drop_capabilities:
                docker_args.extend(["--cap-drop=ALL"])

            if not self.allowed_network:
                docker_args.extend(["--network=none"])

            if self.read_only_filesystem:
                docker_args.extend(["--read-only"])
                # Add writable temp directory
                docker_args.extend(["--tmpfs=/tmp:rw,noexec,nosuid,size=100m"])

            # Add environment variables
            if environment:
                for key, value in environment.items():
                    docker_args.extend(["-e", f"{key}={value}"])

            # Add image and command
            docker_args.extend([self.image, "sh", "-c", command])

            # Execute
            process = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout + 10  # Extra time for Docker overhead
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise SandboxError(f"Docker command timed out after {self.timeout} seconds")

            execution_time = time.time() - start_time

            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "execution_time": execution_time,
                "sandbox_type": "docker",
                "image": self.image
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": execution_time,
                "error": str(e),
                "sandbox_type": "docker"
            }

    async def _execute_podman(self, command: str, working_dir: Path, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute command in Podman container."""
        start_time = time.time()

        try:
            # Check if Podman is available
            if not await self._check_podman():
                raise SandboxError("Podman is not available")

            # Prepare Podman command (similar to Docker but with podman)
            podman_args = [
                "podman", "run",
                "--rm",
                "--interactive",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                f"--workdir=/workspace",
                f"--volume={working_dir}:/workspace"
            ]

            # Security options
            if self.drop_capabilities:
                podman_args.extend(["--cap-drop=ALL"])

            if not self.allowed_network:
                podman_args.extend(["--network=none"])

            if self.read_only_filesystem:
                podman_args.extend(["--read-only"])
                podman_args.extend(["--tmpfs=/tmp:rw,noexec,nosuid,size=100m"])

            # Add environment variables
            if environment:
                for key, value in environment.items():
                    podman_args.extend(["-e", f"{key}={value}"])

            # Add image and command
            podman_args.extend([self.image, "sh", "-c", command])

            # Execute
            process = await asyncio.create_subprocess_exec(
                *podman_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout + 10
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise SandboxError(f"Podman command timed out after {self.timeout} seconds")

            execution_time = time.time() - start_time

            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "execution_time": execution_time,
                "sandbox_type": "podman",
                "image": self.image
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": execution_time,
                "error": str(e),
                "sandbox_type": "podman"
            }

    async def _execute_sandbox_exec(self, command: str, working_dir: Path, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute command using macOS sandbox-exec."""
        start_time = time.time()

        try:
            # Create sandbox profile
            profile = self._create_sandbox_profile(working_dir)

            # Prepare sandbox-exec command
            sandbox_args = [
                "sandbox-exec",
                "-p", profile,
                "sh", "-c", command
            ]

            # Prepare environment
            env = dict(os.environ)
            if environment:
                env.update(environment)

            # Execute
            process = await asyncio.create_subprocess_exec(
                *sandbox_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise SandboxError(f"Sandbox-exec command timed out after {self.timeout} seconds")

            execution_time = time.time() - start_time

            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "execution_time": execution_time,
                "sandbox_type": "sandbox-exec"
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "command": command,
                "directory": str(working_dir),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": execution_time,
                "error": str(e),
                "sandbox_type": "sandbox-exec"
            }

    async def _check_docker(self) -> bool:
        """Check if Docker is available and working."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()
            return process.returncode == 0

        except Exception:
            return False

    async def _check_podman(self) -> bool:
        """Check if Podman is available and working."""
        try:
            process = await asyncio.create_subprocess_exec(
                "podman", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()
            return process.returncode == 0

        except Exception:
            return False

    def _create_sandbox_profile(self, working_dir: Path) -> str:
        """Create a sandbox profile for macOS sandbox-exec."""
        profile = f"""
(version 1)
(deny default)
(allow process-info* (target self))
(allow process-info-pidinfo (target self))
(allow process-info-pidfdinfo (target self))
(allow process-info-pidfileportinfo (target self))
(allow process-info-setcontrol (target self))
(allow process-info-dirtycontrol (target self))
(allow process-info-rusage (target self))

(allow file-read*
    (subpath "/usr/lib")
    (subpath "/usr/share")
    (subpath "/System")
    (subpath "/Library")
    (subpath "{working_dir}")
    (literal "/bin/sh")
    (literal "/usr/bin/env"))

(allow file-write*
    (subpath "{working_dir}")
    (subpath "/tmp"))

(allow file-read-metadata
    (literal "/etc")
    (literal "/var")
    (literal "/")
    (literal "/usr"))

(allow mach-lookup)
(allow ipc-posix-shm-read-data)
(allow ipc-posix-shm-write-data)
"""

        if not self.allowed_network:
            profile += """
(deny network*)
"""
        else:
            profile += """
(allow network-outbound)
(allow network-bind)
"""

        return profile

    async def cleanup(self):
        """Clean up temporary resources."""
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

        self.temp_dirs.clear()

        # Clean up containers (if any were created and not auto-removed)
        for container_id in self.containers:
            try:
                if self.sandbox_type == "docker":
                    await self._stop_docker_container(container_id)
                elif self.sandbox_type == "podman":
                    await self._stop_podman_container(container_id)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup container {container_id}: {e}")

        self.containers.clear()

    async def _stop_docker_container(self, container_id: str):
        """Stop and remove a Docker container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            process = await asyncio.create_subprocess_exec(
                "docker", "rm", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

        except Exception as e:
            self.logger.warning(f"Failed to stop Docker container {container_id}: {e}")

    async def _stop_podman_container(self, container_id: str):
        """Stop and remove a Podman container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "podman", "stop", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            process = await asyncio.create_subprocess_exec(
                "podman", "rm", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

        except Exception as e:
            self.logger.warning(f"Failed to stop Podman container {container_id}: {e}")

    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get information about the sandbox configuration."""
        return {
            "enabled": self.enabled,
            "type": self.sandbox_type,
            "image": self.image,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
            "allow_network": self.allowed_network,
            "read_only_filesystem": self.read_only_filesystem,
            "drop_capabilities": self.drop_capabilities,
            "temp_dirs": len(self.temp_dirs),
            "containers": len(self.containers)
        }

    async def build_sandbox_image(self, dockerfile_content: str, image_name: Optional[str] = None) -> bool:
        """Build a custom sandbox image."""
        if not self.enabled or self.sandbox_type not in ("docker", "podman"):
            return False

        if image_name is None:
            image_name = self.image

        try:
            # Create temporary dockerfile
            temp_dir = Path(tempfile.mkdtemp())
            dockerfile = temp_dir / "Dockerfile"

            with open(dockerfile, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)

            # Build image
            build_cmd = [self.sandbox_type, "build", "-t", image_name, str(temp_dir)]

            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Cleanup
            shutil.rmtree(temp_dir)

            if process.returncode == 0:
                self.logger.info(f"Successfully built sandbox image: {image_name}")
                return True
            else:
                self.logger.error(f"Failed to build sandbox image: {stderr.decode()}")
                return False

        except Exception as e:
            self.logger.error(f"Error building sandbox image: {e}")
            return False