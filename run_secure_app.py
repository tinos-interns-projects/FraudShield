#!/usr/bin/env python3
"""
Secure launcher for the Fraud Detection System
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔒 Starting Secure Fraud Detection System...")
    print("=" * 60)
    
    # Import and run the app
    from app import app
    
    print("✅ Security modules loaded successfully!")
    print("✅ Authentication system initialized!")
    print("✅ CSRF protection enabled!")
    print("✅ Rate limiting configured!")
    print("=" * 60)
    print("🚀 SECURE FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("🔐 Login Page: http://127.0.0.1:5000/login")
    print("🏠 Home Page: http://127.0.0.1:5000/")
    print("📊 Dashboard: http://127.0.0.1:5000/dashboard (requires login)")
    print("🔍 Predictions: http://127.0.0.1:5000/predict (requires login)")
    print("📁 Upload: http://127.0.0.1:5000/upload (requires login)")
    print("=" * 60)
    print("🔑 Demo Credentials:")
    print("   Admin: admin / SecurePassword123!")
    print("   User:  demo / DemoUser456!")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask app
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False  # Disable reloader to avoid issues
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install flask flask-login flask-wtf flask-limiter bcrypt python-dotenv")
    
except Exception as e:
    print(f"❌ Error starting secure app: {e}")
    import traceback
    traceback.print_exc()
