from flask import Flask, render_template
import os

# Create Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

@app.route('/')
def home():
    return '''
    <html>
    <head><title>Fraud Detection System</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>🚀 Fraud Detection System</h1>
        <p>The Flask app is running successfully!</p>
        <ul>
            <li><a href="/dashboard">Dashboard</a></li>
            <li><a href="/predict">Single Prediction</a></li>
            <li><a href="/upload">Batch Upload</a></li>
            <li><a href="/visualizations">Visualizations</a></li>
        </ul>
        <p><strong>Status:</strong> ✅ Server is running on http://127.0.0.1:5000</p>
    </body>
    </html>
    '''

@app.route('/dashboard')
def dashboard():
    try:
        return render_template('dashboard.html')
    except:
        return '<h1>Dashboard</h1><p>Template not found, but route is working!</p>'

@app.route('/predict')
def predict():
    try:
        return render_template('predict.html')
    except:
        return '<h1>Prediction</h1><p>Template not found, but route is working!</p>'

@app.route('/upload')
def upload():
    try:
        return render_template('upload.html')
    except:
        return '<h1>Upload</h1><p>Template not found, but route is working!</p>'

@app.route('/visualizations')
def visualizations():
    try:
        return render_template('visualizations.html')
    except:
        return '<h1>Visualizations</h1><p>Template not found, but route is working!</p>'

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 STARTING FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("📊 Home Page: http://127.0.0.1:5000/")
    print("📊 Dashboard: http://127.0.0.1:5000/dashboard")
    print("🔍 Prediction: http://127.0.0.1:5000/predict")
    print("📁 Upload: http://127.0.0.1:5000/upload")
    print("📈 Visualizations: http://127.0.0.1:5000/visualizations")
    print("=" * 60)
    print("✅ Server starting... Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False
    )
