# api/app.py
from flask import Flask, request, jsonify, render_template, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

from config.settings import config
from models.emotion_classifier import EmotionClassifier
from models.art_generator import ArtGenerator
from models.therapy_recommender import TherapyRecommender
from analysis.progress_tracker import ProgressTracker

app = Flask(__name__)
app.config.from_object(config)

emotion_classifier = EmotionClassifier()
art_generator = ArtGenerator()
therapy_recommender = TherapyRecommender()
progress_tracker = ProgressTracker()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze/artwork', methods=['POST'])
def analyze_artwork():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        user_description = request.form.get('description', '')
        user_id = request.form.get('user_id', str(uuid.uuid4()))
        
        try:
            analysis = emotion_classifier.analyze_artwork(filepath, user_description)
            
            session_data = progress_tracker.record_session(
                user_id=user_id,
                art_analysis=analysis,
                exercise_completed='art_analysis',
                user_feedback=user_description
            )
            
            analysis['session_id'] = session_data['session_id']
            analysis['user_id'] = user_id
            
            return jsonify(analysis)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate/exercise', methods=['POST'])
def generate_exercise():
    data = request.get_json()
    
    exercise_type = data.get('exercise_type', 'mindfulness')
    user_emotion = data.get('user_emotion', 'neutral')
    style_preference = data.get('style_preference', 'abstract')
    
    try:
        result = art_generator.generate_therapeutic_art(
            exercise_type, 
            user_emotion, 
            style_preference
        )
        
        img_io = io.BytesIO()
        result['image'].save(img_io, 'PNG')
        img_io.seek(0)
        
        img_data = base64.b64encode(img_io.getvalue()).decode()
        
        return jsonify({
            'exercise_type': result['exercise_type'],
            'specific_exercise': result['specific_exercise'],
            'therapeutic_intent': result['therapeutic_intent'],
            'prompt_used': result['prompt_used'],
            'generated_image': f"data:image/png;base64,{img_data}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend/therapy', methods=['POST'])
def recommend_therapy():
    data = request.get_json()
    
    user_emotion = data.get('emotion', 'neutral')
    art_style = data.get('art_style', 'abstract')
    user_experience = data.get('experience_level', 'beginner')
    
    try:
        recommendations = therapy_recommender.recommend_exercises(
            user_emotion, 
            art_style, 
            user_experience
        )
        
        return jsonify({
            'recommendations': recommendations,
            'user_profile': {
                'emotion': user_emotion,
                'preferred_style': art_style,
                'experience_level': user_experience
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<user_id>', methods=['GET'])
def get_progress(user_id):
    try:
        progress_report = progress_tracker.generate_progress_report(user_id)
        return jsonify(progress_report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create/plan', methods=['POST'])
def create_plan():
    data = request.get_json()
    
    user_goals = data.get('goals', ['self_expression', 'emotional_awareness'])
    current_emotion = data.get('current_emotion', 'neutral')
    timeline_weeks = data.get('timeline_weeks', 8)
    
    try:
        plan = therapy_recommender.generate_progress_plan(
            user_goals, 
            current_emotion, 
            timeline_weeks
        )
        
        return jsonify(plan)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personalized/prompt', methods=['POST'])
def personalized_prompt():
    data = request.get_json()
    
    user_emotion = data.get('emotion')
    art_style = data.get('art_style')
    therapeutic_goal = data.get('therapeutic_goal')
    
    if not all([user_emotion, art_style, therapeutic_goal]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        prompt = art_generator.generate_personalized_prompt(
            user_emotion, 
            art_style, 
            therapeutic_goal
        )
        
        return jsonify({'personalized_prompt': prompt})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GENERATED_ART_FOLDER'], exist_ok=True)
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Starting MindCanvas Art Therapy Assistant...")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)