# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    IMAGE_UPLOAD_FOLDER = "static/uploads"
    GENERATED_ART_FOLDER = "static/generated"
    MODEL_SAVE_PATH = "trained_models/art_therapy_model.pth"
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    CV_MODEL_NAME = "microsoft/resnet-50"
    NLP_MODEL_NAME = "distilbert-base-uncased"
    GENERATIVE_MODEL = "runwayml/stable-diffusion-v1-5"
    
    EMOTION_LABELS = [
        'joy', 'sadness', 'anger', 'fear', 
        'surprise', 'disgust', 'neutral', 'anxiety',
        'calm', 'confusion', 'hope', 'despair'
    ]
    
    ART_STYLES = [
        'abstract', 'expressionist', 'surreal', 'realistic',
        'minimalist', 'cubist', 'impressionist', 'pop_art'
    ]
    
    THERAPEUTIC_EXERCISES = {
        'mindfulness': ['breathing_circle', 'mandala', 'texture_exploration'],
        'emotional_expression': ['emotion_color_wheel', 'feeling_landscape', 'inner_child'],
        'trauma_recovery': ['safe_space', 'container_exercise', 'body_map'],
        'self_discovery': ['life_tree', 'mask_exercise', 'future_self']
    }

config = Config()