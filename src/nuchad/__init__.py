"""CHADS-VASc Analysis Package for stroke risk prediction in AF patients."""

__version__ = "0.1.0"

from nuchad.analysis import eda, table2, density_ratio_reweighting
from nuchad.visualization import visualize_end_fu
from nuchad.utils import (
    get_project_root,
    get_data_file,
    get_data_path,
    get_results_dir
)

def main() -> None:
    """Main entry point for the package."""
    from .__main__ import main as _main
    _main()
