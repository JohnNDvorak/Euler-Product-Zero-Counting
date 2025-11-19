"""
Path configuration and utilities.

This module manages paths for data, cache, and results directories,
with support for different environments (local, Colab, etc.).
"""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml


class PathConfig:
    """
    Centralized path configuration for the project.
    """

    def __init__(self, config_path: Optional[str] = None, base_dir: Optional[str] = None):
        """
        Initialize path configuration.

        Parameters:
        -----------
        config_path : str, optional
            Path to configuration YAML file
        base_dir : str, optional
            Base directory for the project (overrides config)
        """
        self.base_dir = Path(base_dir) if base_dir else self._find_base_dir()
        self.config_path = config_path or self.base_dir / "config.yaml"
        self._config = self._load_config()

    def _find_base_dir(self) -> Path:
        """Automatically find the base directory."""
        # Start from current directory and search upward
        current = Path.cwd()
        while current != current.parent:
            if (current / "src").exists() and (current / "README.md").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration if no config file found."""
        return {
            'paths': {
                'base': str(self.base_dir),
                'data': str(self.base_dir / "data"),
                'cache': str(self.base_dir / "cache"),
                'results': str(self.base_dir / "results"),
                'figures': str(self.base_dir / "results" / "figures")
            },
            'data': {
                'zeros_file': '10M_zeta_zeros.txt',
                'cache_file': 'cache/prime_cache_1B.pkl'
            },
            'experiments': {
                't_values': [1_000, 10_000, 100_000, 1_000_000],
                'p_max_range': [1e6, 1e9],
                'n_points': 50
            }
        }

    @property
    def paths(self) -> Dict[str, Path]:
        """Get all configured paths as Path objects."""
        paths = {}
        for name, path_str in self._config['paths'].items():
            paths[name] = Path(path_str)
        return paths

    def get_path(self, name: str) -> Path:
        """Get a specific path by name."""
        if name in self.paths:
            return self.paths[name]
        raise ValueError(f"Unknown path name: {name}")

    def ensure_dirs(self):
        """Ensure all configured directories exist."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {path}")

    @property
    def data_dir(self) -> Path:
        """Data directory."""
        return self.get_path('data')

    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self.get_path('cache')

    @property
    def results_dir(self) -> Path:
        """Results directory."""
        return self.get_path('results')

    @property
    def figures_dir(self) -> Path:
        """Figures directory."""
        return self.get_path('figures')

    @property
    def zeros_file(self) -> Path:
        """Riemann zeros file path."""
        # Check in data/raw first, then in root
        file_path = self.data_dir / "raw" / self._config['data']['zeros_file']
        if not file_path.exists():
            file_path = self.base_dir / self._config['data']['zeros_file']
        return file_path

    @property
    def prime_cache_file(self) -> Path:
        """Prime cache file path."""
        return self.cache_dir / self._config['data']['cache_file']


def check_prerequisites(required_files: list, verbose: bool = True) -> bool:
    """
    Check if required files exist.

    Parameters:
    -----------
    required_files : list
        List of file paths to check
    verbose : bool
        Whether to print status

    Returns:
    --------
    bool : True if all files exist
    """
    all_exist = True

    if verbose:
        print("Checking prerequisites:")
        print("-" * 40)

    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()

        if verbose:
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")

        if not exists:
            all_exist = False

    return all_exist