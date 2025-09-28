from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Flask is working!</h1><p>Your fraud detection app should work too.</p>'

if __name__ == '__main__':
    print("Starting test Flask server...")
    print("Visit: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
