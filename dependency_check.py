#!/usr/bin/env python3
"""
Dependency Check and Installation for Generations 7-8-9
Terragon Labs Autonomous SDLC Environment Setup
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def install_package(package_name):
    """Install a Python package using pip."""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies."""
    
    # Required packages for Generations 7-8-9
    required_packages = [
        'numpy',
        'scipy',  # For advanced mathematical operations
    ]
    
    print("🔍 Checking and installing dependencies...")
    
    installation_results = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} already available")
            installation_results.append(True)
        except ImportError:
            print(f"⚠️ {package} not found, attempting installation...")
            result = install_package(package)
            installation_results.append(result)
    
    return all(installation_results)

def fix_typing_import():
    """Fix typing import issues for older Python versions."""
    
    # Check if we can import Complex from typing
    try:
        from typing import Complex
        print("✅ typing.Complex available")
        return True
    except ImportError:
        print("⚠️ typing.Complex not available, will use alternative")
        
        # Create a compatibility fix
        compat_code = '''
# Compatibility fix for typing.Complex
import typing
if not hasattr(typing, 'Complex'):
    typing.Complex = complex
'''
        
        # Write compatibility file
        compat_path = os.path.join('src', 'typing_compat.py')
        os.makedirs('src', exist_ok=True)
        
        with open(compat_path, 'w') as f:
            f.write(compat_code)
        
        print(f"✅ Created compatibility fix: {compat_path}")
        return True

def create_minimal_implementations():
    """Create minimal implementations that don't require heavy dependencies."""
    
    print("🔧 Creating minimal implementations...")
    
    # Update quantum_reasoner.py to remove Complex import
    quantum_file = 'src/neuro_symbolic_law/quantum/quantum_reasoner.py'
    
    if os.path.exists(quantum_file):
        with open(quantum_file, 'r') as f:
            content = f.read()
        
        # Replace Complex import
        content = content.replace('from typing import Dict, List, Any, Optional, Tuple, Set, Complex, Union',
                                'from typing import Dict, List, Any, Optional, Tuple, Set, Union')
        content = content.replace(', Complex', '')
        content = content.replace('Complex', 'complex')
        
        with open(quantum_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {quantum_file}")
    
    return True

def run_environment_setup():
    """Run complete environment setup."""
    
    print("🚀 TERRAGON LABS ENVIRONMENT SETUP")
    print("=" * 50)
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Dependency Installation", check_and_install_dependencies),
        ("Typing Compatibility", fix_typing_import),
        ("Implementation Fixes", create_minimal_implementations)
    ]
    
    results = []
    
    for step_name, step_func in setup_steps:
        print(f"\n🎯 {step_name}")
        print("-" * 30)
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 ENVIRONMENT SETUP SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for step_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{step_name:.<30} {status}")
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("\n✅ ENVIRONMENT READY")
        print("You can now run quality gates validation")
        return True
    else:
        print("\n❌ ENVIRONMENT SETUP INCOMPLETE")
        print("Some dependencies may be missing")
        return False

if __name__ == "__main__":
    success = run_environment_setup()
    
    if success:
        print("\n🚀 Next step: Run quality gates validation")
        print("   python3 simple_quality_gates.py")
    
    exit(0 if success else 1)