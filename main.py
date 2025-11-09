# main.py
from api.app import app
from config.settings import config
import os

if __name__ == '__main__':
    os.makedirs(config.IMAGE_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.GENERATED_ART_FOLDER, exist_ok=True)
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("MindCanvas AI Art Therapy Assistant Starting...")
    print("Web interface: http://localhost:5000")
    print("API endpoints ready for integration")
    
    app.run(debug=True, host='0.0.0.0', port=5000)