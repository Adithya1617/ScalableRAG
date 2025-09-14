#!/usr/bin/env python3
"""
Fix dependency issues for RAG system
"""

import subprocess
import sys

def main():
    """Fix common dependency issues"""
    print("🔧 Fixing dependency issues...")
    
    try:
        # First, reinstall numpy specifically
        print("📦 Reinstalling numpy...")
        subprocess.run([
            "uv", "pip", "uninstall", "numpy", "-y"
        ], capture_output=True)
        
        subprocess.run([
            "uv", "pip", "install", "numpy>=1.21.0"
        ], check=True)
        print("✅ numpy fixed")
        
        # Then reinstall transformers
        print("📦 Reinstalling transformers...")
        subprocess.run([
            "uv", "pip", "uninstall", "transformers", "-y"
        ], capture_output=True)
        
        subprocess.run([
            "uv", "pip", "install", "transformers"
        ], check=True)
        print("✅ transformers fixed")
        
        # Finally, reinstall all requirements
        print("📦 Reinstalling all requirements...")
        subprocess.run([
            "uv", "pip", "install", "-r", "requirements.txt", "--force-reinstall"
        ], check=True)
        print("✅ All dependencies fixed")
        
        print("\n🎉 Dependencies fixed! You can now run:")
        print("   python run_backend.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fixing dependencies: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
