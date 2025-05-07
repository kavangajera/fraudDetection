from flask import Flask, jsonify
from flask_cors import CORS
from routes.train import train_bp
from routes.test import test_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Register Blueprints under "/api"
app.register_blueprint(train_bp, url_prefix='/api/train')
app.register_blueprint(test_bp, url_prefix='/api/test')

# Root route
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to watchdog"}), 200

if __name__ == '__main__':
    app.run(debug=True)
