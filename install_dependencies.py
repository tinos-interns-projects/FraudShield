#!/usr/bin/env python3
"""
Install all required dependencies for the Fraud Detection System
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_package(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Install all required packages"""
    print("🔧 Installing Fraud Detection System Dependencies")
    print("=" * 60)
    
    # Core packages
    packages = [
        "flask",
        "flask-login", 
        "flask-wtf",
        "flask-limiter",
        "bcrypt",
        "email-validator",
        "pandas",
        "numpy", 
        "scikit-learn",
        "xgboost",
        "plotly",
        "werkzeug",
        "openpyxl",
        "PyPDF2",
        "tabula-py",
        "pdfplumber", 
        "reportlab",
        "python-dotenv",
        "requests",
        "dnspython"
    ]
    
    installed_count = 0
    failed_count = 0
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            installed_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary:")
    print(f"✅ Successfully installed: {installed_count}")
    print(f"❌ Failed to install: {failed_count}")
    
    if failed_count == 0:
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run the Fraud Detection System:")
        print("   python run_secure_app.py")
    else:
        print(f"\n⚠️  {failed_count} packages failed to install.")
        print("Please install them manually or check your internet connection.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
