"""
Checkpointing system for Gemini CLI
Provides undo/restore functionality for tool operations
"""

import os
import json
import shutil
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import tempfile
import zipfile
import aiofiles

from .exceptions import GeminiCLIError, CheckpointError


class GitCheckpointBackend:
    """Git-based checkpoint backend using shadow repository."""

    def __init__(self, project_path: Path, checkpoint_dir: Path):
        self.project_path = project_path
        self.checkpoint_dir = checkpoint_dir
        self.shadow_repo = checkpoint_dir / "shadow_repo"
        self.logger = logging.getLogger(f"{__name__}.GitCheckpointBackend")

        # Initialize shadow repository
        asyncio.create_task(self._initialize_shadow_repo())

    async def _initialize_shadow_repo(self):
        """Initialize the shadow git repository."""
        try:
            if not self.shadow_repo.exists():
                self.shadow_repo.mkdir(parents=True)

                # Initialize git repo
                await self._run_git_command(["init"], cwd=self.shadow_repo)

                # Configure git
                await self._run_git_command([
                    "config", "user.name", "Gemini CLI Checkpoint System"
                ], cwd=self.shadow_repo)
                await self._run_git_command([
                    "config", "user.email", "checkpoint@gemini-cli.local"
                ], cwd=self.shadow_repo)

                self.logger.info(f"Initialized shadow repository: {self.shadow_repo}")

        except Exception as e:
            self.logger.error(f"Failed to initialize shadow repository: {e}")

    async def create_snapshot(self, checkpoint_id: str, files: List[Path]) -> bool:
        """Create a git snapshot of specified files."""
        try:
            # Copy files to shadow repo
            for file_path in files:
                if not file_path.exists():
                    continue

                # Calculate relative path from project root
                try:
                    rel_path = file_path.relative_to(self.project_path)
                except ValueError:
                    # File is outside project, skip
                    continue

                # Create target path in shadow repo
                target_path = self.shadow_repo / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(file_path, target_path)

            # Add all files to git
            await self._run_git_command(["add", "."], cwd=self.shadow_repo)

            # Create commit
            commit_message = f"Checkpoint: {checkpoint_id}"
            await self._run_git_command([
                "commit", "-m", commit_message, "--allow-empty"
            ], cwd=self.shadow_repo)

            # Tag the commit
            await self._run_git_command([
                "tag", checkpoint_id
            ], cwd=self.shadow_repo)

            self.logger.debug(f"Created git snapshot: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create git snapshot {checkpoint_id}: {e}")
            return False

    async def restore_snapshot(self, checkpoint_id: str) -> bool:
        """Restore files from a git snapshot."""
        try:
            # Check if tag exists
            result = await self._run_git_command([
                "tag", "-l", checkpoint_id
            ], cwd=self.shadow_repo, capture_output=True)

            if checkpoint_id not in result.stdout:
                self.logger.error(f"Checkpoint tag not found: {checkpoint_id}")
                return False

            # Reset to the tagged commit
            await self._run_git_command([
                "reset", "--hard", checkpoint_id
            ], cwd=self.shadow_repo)

            # Copy files back to project
            await self._copy_files_to_project()

            self.logger.info(f"Restored git snapshot: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore git snapshot {checkpoint_id}: {e}")
            return False

    async def list_snapshots(self) -> List[Tuple[str, str, datetime]]:
        """List available git snapshots."""
        try:
            # Get tags with commit info
            result = await self._run_git_command([
                "for-each-ref", "--format=%(refname:short) %(objectname:short) %(creatordate:iso)",
                "refs/tags"
            ], cwd=self.shadow_repo, capture_output=True)

            snapshots = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    tag_name = parts[0]
                    commit_hash = parts[1]
                    date_str = ' '.join(parts[2:])

                    try:
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        snapshots.append((tag_name, commit_hash, date))
                    except ValueError:
                        continue

            return sorted(snapshots, key=lambda x: x[2], reverse=True)

        except Exception as e:
            self.logger.error(f"Failed to list git snapshots: {e}")
            return []

    async def delete_snapshot(self, checkpoint_id: str) -> bool:
        """Delete a git snapshot."""
        try:
            # Delete tag
            await self._run_git_command([
                "tag", "-d", checkpoint_id
            ], cwd=self.shadow_repo)

            self.logger.debug(f"Deleted git snapshot: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete git snapshot {checkpoint_id}: {e}")
            return False

    async def _copy_files_to_project(self):
        """Copy files from shadow repo back to project."""
        for root, dirs, files in os.walk(self.shadow_repo):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')

            for file in files:
                source_path = Path(root) / file

                # Calculate relative path
                try:
                    rel_path = source_path.relative_to(self.shadow_repo)
                except ValueError:
                    continue

                # Target path in project
                target_path = self.project_path / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(source_path, target_path)

    async def _run_git_command(self, args: List[str], cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args

        if capture_output:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return type('CompletedProcess', (), {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            })()
        else:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            await process.wait()
            return type('CompletedProcess', (), {'returncode': process.returncode})()


class FileCheckpointBackend:
    """File-based checkpoint backend using zip archives."""

    def __init__(self, project_path: Path, checkpoint_dir: Path):
        self.project_path = project_path
        self.checkpoint_dir = checkpoint_dir
        self.snapshots_dir = checkpoint_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.FileCheckpointBackend")

    async def create_snapshot(self, checkpoint_id: str, files: List[Path]) -> bool:
        """Create a file snapshot using zip archive."""
        try:
            snapshot_file = self.snapshots_dir / f"{checkpoint_id}.zip"

            with zipfile.ZipFile(snapshot_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if not file_path.exists():
                        continue

                    try:
                        # Calculate relative path
                        rel_path = file_path.relative_to(self.project_path)
                        zipf.write(file_path, rel_path)
                    except ValueError:
                        # File outside project, skip
                        continue

            self.logger.debug(f"Created file snapshot: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create file snapshot {checkpoint_id}: {e}")
            return False

    async def restore_snapshot(self, checkpoint_id: str) -> bool:
        """Restore files from a zip snapshot."""
        try:
            snapshot_file = self.snapshots_dir / f"{checkpoint_id}.zip"

            if not snapshot_file.exists():
                self.logger.error(f"Snapshot file not found: {snapshot_file}")
                return False

            with zipfile.ZipFile(snapshot_file, 'r') as zipf:
                # Extract all files to project directory
                zipf.extractall(self.project_path)

            self.logger.info(f"Restored file snapshot: {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore file snapshot {checkpoint_id}: {e}")
            return False

    async def list_snapshots(self) -> List[Tuple[str, str, datetime]]:
        """List available file snapshots."""
        try:
            snapshots = []

            for snapshot_file in self.snapshots_dir.glob("*.zip"):
                checkpoint_id = snapshot_file.stem
                file_stats = snapshot_file.stat()
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                file_hash = self._calculate_file_hash(snapshot_file)

                snapshots.append((checkpoint_id, file_hash[:8], modified_time))

            return sorted(snapshots, key=lambda x: x[2], reverse=True)

        except Exception as e:
            self.logger.error(f"Failed to list file snapshots: {e}")
            return []

    async def delete_snapshot(self, checkpoint_id: str) -> bool:
        """Delete a file snapshot."""
        try:
            snapshot_file = self.snapshots_dir / f"{checkpoint_id}.zip"

            if snapshot_file.exists():
                snapshot_file.unlink()
                self.logger.debug(f"Deleted file snapshot: {checkpoint_id}")
                return True
            else:
                self.logger.warning(f"Snapshot file not found: {checkpoint_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete file snapshot {checkpoint_id}: {e}")
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "unknown"


class CheckpointManager:
    """Manages checkpoints for undo/restore functionality."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_config = config.get("checkpointing", {})
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.enabled = self.checkpoint_config.get("enabled", False)
        self.max_checkpoints = self.checkpoint_config.get("maxCheckpoints", 10)
        self.backend_type = self.checkpoint_config.get("backend", "git")  # "git" or "file"

        # Paths
        self.project_path = Path.cwd()
        self.project_hash = self._calculate_project_hash()
        self.checkpoint_dir = Path.home() / ".gemini" / "checkpoints" / self.project_hash
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Backend
        if self.backend_type == "git":
            self.backend = GitCheckpointBackend(self.project_path, self.checkpoint_dir)
        else:
            self.backend = FileCheckpointBackend(self.project_path, self.checkpoint_dir)

        # Checkpoint history
        self.checkpoint_history: List[Dict[str, Any]] = []

    async def create_checkpoint(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        original_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Create a checkpoint before tool execution."""
        if not self.enabled:
            return ""

        try:
            # Generate checkpoint ID
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S_%fZ")
            checkpoint_id = f"{timestamp}-{tool_name}"

            # Determine files to checkpoint
            files_to_checkpoint = await self._determine_files_to_checkpoint(tool_name, tool_args)

            # Create snapshot
            success = await self.backend.create_snapshot(checkpoint_id, files_to_checkpoint)

            if not success:
                raise CheckpointError("Failed to create snapshot")

            # Save checkpoint metadata
            checkpoint_data = {
                "id": checkpoint_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "original_prompt": original_prompt,
                "conversation_history": conversation_history or [],
                "created_at": datetime.now().isoformat(),
                "files": [str(f) for f in files_to_checkpoint],
                "project_path": str(self.project_path)
            }

            await self._save_checkpoint_metadata(checkpoint_id, checkpoint_data)

            # Add to history
            self.checkpoint_history.append(checkpoint_data)

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

            self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise CheckpointError(f"Checkpoint creation failed: {e}")

    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore a checkpoint."""
        if not self.enabled:
            raise CheckpointError("Checkpointing is not enabled")

        try:
            # Load checkpoint metadata
            checkpoint_data = await self._load_checkpoint_metadata(checkpoint_id)

            if not checkpoint_data:
                raise CheckpointError(f"Checkpoint not found: {checkpoint_id}")

            # Restore snapshot
            success = await self.backend.restore_snapshot(checkpoint_id)

            if not success:
                raise CheckpointError("Failed to restore snapshot")

            self.logger.info(f"Restored checkpoint: {checkpoint_id}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            raise CheckpointError(f"Checkpoint restore failed: {e}")

    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        if not self.enabled:
            return []

        try:
            # Get snapshots from backend
            snapshots = await self.backend.list_snapshots()

            # Load metadata for each checkpoint
            checkpoints = []
            for checkpoint_id, commit_hash, created_at in snapshots:
                metadata = await self._load_checkpoint_metadata(checkpoint_id)
                if metadata:
                    metadata["commit_hash"] = commit_hash
                    metadata["created_at"] = created_at.isoformat()
                    checkpoints.append(metadata)

            return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        if not self.enabled:
            return False

        try:
            # Delete snapshot
            success = await self.backend.delete_snapshot(checkpoint_id)

            if success:
                # Delete metadata
                await self._delete_checkpoint_metadata(checkpoint_id)

                # Remove from history
                self.checkpoint_history = [
                    cp for cp in self.checkpoint_history if cp["id"] != checkpoint_id
                ]

                self.logger.info(f"Deleted checkpoint: {checkpoint_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def _determine_files_to_checkpoint(self, tool_name: str, tool_args: Dict[str, Any]) -> List[Path]:
        """Determine which files should be included in the checkpoint."""
        files = set()

        # Tool-specific file detection
        if tool_name == "write_file":
            file_path = tool_args.get("file_path")
            if file_path:
                files.add(Path(file_path))

        elif tool_name == "replace":
            file_path = tool_args.get("file_path")
            if file_path:
                files.add(Path(file_path))

        elif tool_name == "run_shell_command":
            # For shell commands, checkpoint entire project (or commonly modified files)
            common_files = [
                "package.json", "package-lock.json", "requirements.txt", "Pipfile",
                "Cargo.toml", "go.mod", "composer.json", "pom.xml", "build.gradle",
                "Makefile", "CMakeLists.txt", ".gitignore", "README.md"
            ]

            for filename in common_files:
                file_path = self.project_path / filename
                if file_path.exists():
                    files.add(file_path)

        # Always include certain project files if they exist
        important_files = [
            ".gemini/settings.json",
            "GEMINI.md",
            ".env",
            "config.json",
            "config.yaml"
        ]

        for filename in important_files:
            file_path = self.project_path / filename
            if file_path.exists():
                files.add(file_path)

        # Convert to list and resolve paths
        resolved_files = []
        for file_path in files:
            try:
                resolved_path = file_path.resolve()
                if resolved_path.exists() and resolved_path.is_file():
                    resolved_files.append(resolved_path)
            except Exception:
                continue

        return resolved_files

    async def _save_checkpoint_metadata(self, checkpoint_id: str, metadata: Dict[str, Any]):
        """Save checkpoint metadata to file."""
        metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))

    async def _load_checkpoint_metadata(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint metadata from file."""
        metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not metadata_file.exists():
            return None

        try:
            async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint metadata {checkpoint_id}: {e}")
            return None

    async def _delete_checkpoint_metadata(self, checkpoint_id: str):
        """Delete checkpoint metadata file."""
        metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if metadata_file.exists():
            metadata_file.unlink()

    async def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to stay within limits."""
        try:
            checkpoints = await self.list_checkpoints()

            if len(checkpoints) > self.max_checkpoints:
                # Sort by creation time (oldest first)
                sorted_checkpoints = sorted(checkpoints, key=lambda x: x["created_at"])

                # Delete oldest checkpoints
                to_delete = sorted_checkpoints[:-self.max_checkpoints]
                for checkpoint in to_delete:
                    await self.delete_checkpoint(checkpoint["id"])

                self.logger.debug(f"Cleaned up {len(to_delete)} old checkpoints")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")

    def _calculate_project_hash(self) -> str:
        """Calculate a hash for the current project path."""
        path_str = str(self.project_path.resolve())
        return hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:16]

    def _load_metadata(self) -> Dict[str, Any]:
        """Load general metadata."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_metadata(self):
        """Save general metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        return {
            "enabled": self.enabled,
            "backend": self.backend_type,
            "max_checkpoints": self.max_checkpoints,
            "checkpoint_dir": str(self.checkpoint_dir),
            "project_hash": self.project_hash,
            "total_checkpoints": len(self.checkpoint_history)
        }

    async def export_checkpoint(self, checkpoint_id: str, output_path: Path) -> bool:
        """Export a checkpoint to a file."""
        try:
            checkpoint_data = await self._load_checkpoint_metadata(checkpoint_id)
            if not checkpoint_data:
                return False

            # Create export archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata
                zipf.writestr("metadata.json", json.dumps(checkpoint_data, indent=2))

                # Add snapshot if it exists (for file backend)
                if self.backend_type == "file":
                    snapshot_file = self.checkpoint_dir / "snapshots" / f"{checkpoint_id}.zip"
                    if snapshot_file.exists():
                        zipf.write(snapshot_file, "snapshot.zip")

            self.logger.info(f"Exported checkpoint {checkpoint_id} to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export checkpoint {checkpoint_id}: {e}")
            return False

    async def import_checkpoint(self, import_path: Path) -> Optional[str]:
        """Import a checkpoint from a file."""
        try:
            if not import_path.exists():
                return None

            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Extract metadata
                metadata_content = zipf.read("metadata.json").decode('utf-8')
                metadata = json.loads(metadata_content)

                checkpoint_id = metadata["id"]

                # Save metadata
                await self._save_checkpoint_metadata(checkpoint_id, metadata)

                # Import snapshot if present
                if "snapshot.zip" in zipf.namelist() and self.backend_type == "file":
                    snapshot_content = zipf.read("snapshot.zip")
                    snapshot_file = self.checkpoint_dir / "snapshots" / f"{checkpoint_id}.zip"
                    snapshot_file.parent.mkdir(exist_ok=True)

                    with open(snapshot_file, 'wb') as f:
                        f.write(snapshot_content)

            self.logger.info(f"Imported checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            self.logger.error(f"Failed to import checkpoint from {import_path}: {e}")
            return None