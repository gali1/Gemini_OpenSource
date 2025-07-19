"""
Version information for Gemini CLI
Provides version strings and build information
"""

import sys
import platform
from datetime import datetime
from typing import Dict, Any


# Version information
__version__ = "1.0.0"
__build_date__ = "2024-01-01"
__build_number__ = "1"


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> Dict[str, Any]:
    """Get detailed version information."""
    return {
        "version": __version__,
        "build_date": __build_date__,
        "build_number": __build_number__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }


def get_build_info() -> Dict[str, Any]:
    """Get build and environment information."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
    except ImportError:
        torch_version = "Not installed"
        cuda_available = False
        cuda_version = None

    return {
        "gemini_cli_version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "hostname": platform.node(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "build_date": __build_date__,
        "build_number": __build_number__,
    }


def format_version_info() -> str:
    """Format version information as a string."""
    info = get_version_info()
    return f"""Gemini CLI {info['version']}
Build: {info['build_number']} ({info['build_date']})
Python: {info['python_version']}
Platform: {info['platform']} ({info['architecture']})"""


def format_build_info() -> str:
    """Format build information as a string."""
    info = get_build_info()

    lines = [
        f"Gemini CLI Version: {info['gemini_cli_version']}",
        f"Python: {info['python_version']} ({info['python_implementation']})",
        f"Platform: {info['platform']}",
        f"Architecture: {info['architecture']}",
        f"Processor: {info['processor']}",
        f"Hostname: {info['hostname']}",
        f"PyTorch: {info['torch_version']}",
        f"CUDA Available: {info['cuda_available']}",
    ]

    if info['cuda_version']:
        lines.append(f"CUDA Version: {info['cuda_version']}")

    lines.extend([
        f"Build Date: {info['build_date']}",
        f"Build Number: {info['build_number']}"
    ])

    return "\n".join(lines)


def check_compatibility() -> Dict[str, Any]:
    """Check system compatibility."""
    issues = []
    warnings = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher required")
    elif sys.version_info < (3, 9):
        warnings.append("Python 3.9+ recommended for best performance")

    # Check PyTorch availability
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available - using CPU mode")
    except ImportError:
        issues.append("PyTorch not installed")

    # Check required packages
    required_packages = [
        "torch",
        "einops",
        "aiofiles",
        "aiohttp",
        "sentencepiece"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        issues.append(f"Missing required packages: {', '.join(missing_packages)}")

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "system_info": get_build_info()
    }


def format_compatibility_report() -> str:
    """Format compatibility check results."""
    compat = check_compatibility()

    lines = ["Gemini CLI Compatibility Check"]
    lines.append("=" * 35)

    if compat["compatible"]:
        lines.append("✓ System is compatible")
    else:
        lines.append("✗ Compatibility issues found")

    if compat["issues"]:
        lines.append("\nIssues:")
        for issue in compat["issues"]:
            lines.append(f"  ✗ {issue}")

    if compat["warnings"]:
        lines.append("\nWarnings:")
        for warning in compat["warnings"]:
            lines.append(f"  ⚠ {warning}")

    lines.append(f"\nSystem Information:")
    info = compat["system_info"]
    lines.append(f"  Python: {info['python_version']}")
    lines.append(f"  Platform: {info['platform']}")
    lines.append(f"  PyTorch: {info['torch_version']}")
    lines.append(f"  CUDA: {'Available' if info['cuda_available'] else 'Not available'}")

    return "\n".join(lines)