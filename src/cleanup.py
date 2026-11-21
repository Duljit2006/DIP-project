"""
src/cleanup.py - Automatic Cache and Previous Results Cleaner
-------------------------------------------------------------
Utility module to immediately wipe previous evaluation results and
Python bytecode caches (__pycache__, *.pyc) across the project workspace.
"""

import os
import sys
import shutil
import stat
from pathlib import Path


def _handle_remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree on Windows if files/folders are read-only.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def get_project_root() -> Path:
    """
    Resolves the absolute root directory of the project.
    """
    return Path(__file__).resolve().parent.parent


def clean_cache(project_root: Path = None, verbose: bool = True) -> int:
    """
    Finds and deletes all '__pycache__' directories and '*.pyc' / '*.pyo' files
    in the project workspace, excluding virtual environments and git internals.

    Args:
        project_root (Path, optional): Root folder of the project.
        verbose (bool): Whether to print cleanup summary.

    Returns:
        int: Number of cache directories and files removed.
    """
    # Prevent Python from writing bytecode during this process
    sys.dont_write_bytecode = True

    if project_root is None:
        project_root = get_project_root()

    project_root = Path(project_root).resolve()
    ignored_dirs = {".venv", "venv", "env", ".git", ".idea", ".vscode"}

    removed_count = 0

    for root, dirs, files in os.walk(project_root, topdown=True):
        # Prune search to completely ignore virtual environments and git
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        # 1. Remove __pycache__ and .pytest_cache directories
        for d in list(dirs):
            if d in ("__pycache__", ".pytest_cache"):
                dir_path = os.path.join(root, d)
                try:
                    shutil.rmtree(dir_path, onerror=_handle_remove_readonly)
                    removed_count += 1
                except Exception as e:
                    if verbose:
                        print(f"[CLEANUP WARNING] Could not delete {dir_path}: {e}")
                dirs.remove(d)

        # 2. Remove any loose .pyc or .pyo files
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                file_path = os.path.join(root, f)
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                    removed_count += 1
                except Exception:
                    pass

    if verbose and removed_count > 0:
        print(f"[CLEANUP] Deleted {removed_count} bytecode cache item(s) (__pycache__, *.pyc).")

    return removed_count


def clean_results(results_dir: Path = None, project_root: Path = None, verbose: bool = True) -> int:
    """
    Immediately deletes all previous output files (plots, text reports, heatmaps)
    from the results directory, strictly preserving .gitkeep so Git tracking remains intact.

    Args:
        results_dir (Path, optional): Directory containing results.
        project_root (Path, optional): Project root directory.
        verbose (bool): Whether to print cleanup summary.

    Returns:
        int: Number of result files/folders removed.
    """
    if project_root is None:
        project_root = get_project_root()

    if results_dir is None:
        results_dir = Path(project_root) / "results"
    else:
        results_dir = Path(results_dir)

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    removed_count = 0
    for item in results_dir.iterdir():
        # Strictly preserve .gitkeep
        if item.name == ".gitkeep":
            continue

        try:
            if item.is_file() or item.is_symlink():
                os.chmod(item, stat.S_IWRITE)
                item.unlink(missing_ok=True)
                removed_count += 1
            elif item.is_dir():
                shutil.rmtree(item, onerror=_handle_remove_readonly)
                removed_count += 1
        except Exception as e:
            if verbose:
                print(f"[CLEANUP WARNING] Could not delete {item}: {e}")

    # Ensure .gitkeep always exists
    gitkeep_file = results_dir / ".gitkeep"
    if not gitkeep_file.exists():
        try:
            gitkeep_file.touch()
        except Exception:
            pass

    if verbose and removed_count > 0:
        print(f"[CLEANUP] Cleared {removed_count} previous output file(s) from results/ (preserved .gitkeep).")

    return removed_count


def clean_all(clear_results: bool = True, project_root: Path = None, verbose: bool = True):
    """
    Comprehensive cleanup:
    1. Disables bytecode compilation (sys.dont_write_bytecode = True).
    2. Deletes previous outputs in results/ (preserving .gitkeep).
    3. Deletes project __pycache__ and *.pyc files.
    4. Registers an exit handler to ensure cache is wiped when the process finishes.
    """
    # Disable bytecode creation
    sys.dont_write_bytecode = True

    if project_root is None:
        project_root = get_project_root()

    if clear_results:
        clean_results(project_root=project_root, verbose=verbose)

    clean_cache(project_root=project_root, verbose=verbose)

    # Register exit cleanup
    import atexit
    atexit.register(clean_cache, project_root, False)


if __name__ == "__main__":
    print("=" * 60)
    print("  MANUAL WORKSPACE CLEANUP UTILITY")
    print("=" * 60)
    clean_all(clear_results=True, verbose=True)
    print("[SUCCESS] Cleanup finished successfully!\n")
