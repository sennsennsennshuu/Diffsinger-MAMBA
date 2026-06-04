"""
SSM Optimizer Installation Script

This script installs the SSMOptimizer.dll to OpenUtau Dependencies directory
and sets up the necessary configuration.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_dll_source():
    """Find the built DLL in common locations."""
    cpp_dir = Path(__file__).parent / "cpp"
    
    # Check build directories
    possible_paths = [
        cpp_dir / "build" / "Release" / "SSMOptimizer.dll",
        cpp_dir / "build" / "SSMOptimizer.dll",
        cpp_dir / "x64" / "Release" / "SSMOptimizer.dll",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def install_to_openutau(dll_path: Path, header_path: Path, force: bool = False):
    """Install DLL and header to OpenUtau Dependencies."""
    dest_dir = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM")
    
    # Create directory if needed
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy DLL
    dll_dest = dest_dir / "SSMOptimizer.dll"
    if dll_dest.exists() and not force:
        response = input(f"{dll_dest} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return False
    
    shutil.copy2(dll_path, dll_dest)
    print(f"Installed: {dll_dest}")
    
    # Copy header
    if header_path.exists():
        header_dest = dest_dir / "ssm_optimizer.h"
        shutil.copy2(header_path, header_dest)
        print(f"Installed: {header_dest}")
    
    return True


def create_config_file():
    """Create configuration file for OpenUtau integration."""
    dest_dir = Path("C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM")
    config_file = dest_dir / "ssm_config.json"
    
    config = {
        "version": "1.0.0",
        "enabled": True,
        "use_simd": True,
        "use_openmp": True,
        "chunk_size": 64,
        "num_threads": 0,  # 0 = use all available
        "state_cache_size": 100,
        "description": "SSM Optimizer for DiffSinger - High-performance selective scan"
    }
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created: {config_file}")


def verify_installation():
    """Verify the installation by loading the DLL."""
    try:
        from ssm_optimizer_ctypes import SSMOptimizer, is_available
        
        if not is_available():
            print("ERROR: SSMOptimizer.dll not found after installation!")
            return False
        
        optimizer = SSMOptimizer()
        print(f"SSM Optimizer version: {optimizer.version}")
        print(f"Default config: SIMD={optimizer.get_default_config().use_simd}, "
              f"OpenMP={optimizer.get_default_config().use_openmp}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to load SSMOptimizer.dll: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install SSM Optimizer for OpenUtau")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Overwrite existing files without prompting")
    parser.add_argument("--dll", type=Path, 
                        help="Path to SSMOptimizer.dll (auto-detect if not specified)")
    parser.add_argument("--verify", "-v", action="store_true",
                        help="Verify installation after copying")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SSM Optimizer Installation")
    print("=" * 60)
    print()
    
    # Find DLL
    if args.dll:
        dll_path = args.dll
    else:
        dll_path = find_dll_source()
    
    if dll_path is None or not dll_path.exists():
        print("ERROR: SSMOptimizer.dll not found!")
        print()
        print("Please build the C++ library first:")
        print("  1. Open 'Developer Command Prompt for VS 2022'")
        print("  2. cd ssm_optimizer/cpp")
        print("  3. build.bat")
        print()
        return 1
    
    print(f"Found DLL: {dll_path}")
    
    # Find header
    header_path = dll_path.parent.parent / "ssm_optimizer.h"
    if not header_path.exists():
        header_path = Path(__file__).parent / "cpp" / "ssm_optimizer.h"
    
    # Install
    print()
    print(f"Installing to: C:/Users/Asus/Documents/OpenUtau/Dependencies/SSM/")
    print()
    
    if install_to_openutau(dll_path, header_path, args.force):
        create_config_file()
        
        print()
        print("=" * 60)
        print("Installation completed successfully!")
        print("=" * 60)
        
        if args.verify:
            print()
            print("Verifying installation...")
            if verify_installation():
                print("Verification passed!")
            else:
                print("Verification failed!")
                return 1
        
        print()
        print("Next steps:")
        print("  1. Restart OpenUtau if it's currently running")
        print("  2. Load your SSM voicebank and test performance")
        print("  3. Check OpenUtau logs for SSM Optimizer messages")
        
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
