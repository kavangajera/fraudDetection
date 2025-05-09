from flask import Flask, jsonify
from flask_cors import CORS
from routes.train import train_bp
from routes.test import test_bp
from routes.dummy_bank import bank_bp
from werkzeug.middleware.proxy_fix import ProxyFix
import os

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all domains

# Register Blueprints under "/api"
app.register_blueprint(train_bp, url_prefix='/api/train')
app.register_blueprint(test_bp, url_prefix='/api/test')
app.register_blueprint(bank_bp, url_prefix='/api/bank')

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Root route
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to watchdog"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
