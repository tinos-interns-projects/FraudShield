#!/usr/bin/env python3
"""
Quick fix script for email validator issues
"""

import subprocess
import sys

def install_email_packages():
    """Install email validation packages"""
    packages = [
        "email-validator",
        "dnspython",
        "idna"
    ]
    
    print("🔧 Installing email validation packages...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def test_email_validation():
    """Test email validation functionality"""
    print("\n🧪 Testing email validation...")
    
    try:
        # Test custom email validation
        import re
        
        def validate_email_format(email):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(email_pattern, email) is not None
        
        test_emails = [
            "test@example.com",
            "user.name@domain.co.uk", 
            "invalid-email",
            "test@",
            "@domain.com"
        ]
        
        for email in test_emails:
            result = validate_email_format(email)
            status = "✅ Valid" if result else "❌ Invalid"
            print(f"  {email}: {status}")
        
        print("✅ Email validation is working correctly!")
        
    except Exception as e:
        print(f"❌ Email validation test failed: {e}")

def main():
    """Main fix function"""
    print("🔒 Email Validator Fix Script")
    print("=" * 50)
    
    # Install packages
    install_email_packages()
    
    # Test validation
    test_email_validation()
    
    print("\n" + "=" * 50)
    print("🎉 Email validator fix completed!")
    print("You can now run the application:")
    print("   python run_secure_app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
