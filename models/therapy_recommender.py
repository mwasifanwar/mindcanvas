# models/therapy_recommender.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from config.settings import config

class TherapyRecommender:
    def __init__(self):
        self.exercise_database = self._initialize_exercise_database()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._fit_vectorizer()
        
        openai.api_key = config.OPENAI_API_KEY
    
    def _initialize_exercise_database(self):
        exercises = []
        
        for category, sub_exercises in config.THERAPEUTIC_EXERCISES.items():
            for exercise in sub_exercises:
                exercise_data = {
                    'category': category,
                    'exercise': exercise,
                    'description': self._get_exercise_description(category, exercise),
                    'suitable_emotions': self._get_suitable_emotions(category, exercise),
                    'difficulty': np.random.choice(['beginner', 'intermediate', 'advanced']),
                    'duration': f"{np.random.randint(15, 90)} minutes",
                    'materials': self._get_required_materials(category)
                }
                exercises.append(exercise_data)
        
        return pd.DataFrame(exercises)
    
    def _get_exercise_description(self, category, exercise):
        descriptions = {
            'mindfulness': {
                'breathing_circle': "Create a circular artwork while focusing on your breath, allowing the rhythm to guide your marks",
                'mandala': "Draw or paint a symmetrical mandala to promote focus and inner peace",
                'texture_exploration': "Experiment with different materials and textures to enhance sensory awareness"
            },
            'emotional_expression': {
                'emotion_color_wheel': "Assign colors to different emotions and create a wheel showing your current emotional landscape",
                'feeling_landscape': "Create an abstract landscape that represents your inner emotional world",
                'inner_child': "Make art from the perspective of your younger self, using playful and spontaneous expression"
            },
            'trauma_recovery': {
                'safe_space': "Visualize and create an image of a place where you feel completely safe and protected",
                'container_exercise': "Design a symbolic container that can hold difficult emotions or memories",
                'body_map': "Create a body outline and use colors/shapes to represent physical sensations and emotions"
            },
            'self_discovery': {
                'life_tree': "Draw a tree that represents your life, with roots for your past and branches for your future",
                'mask_exercise': "Create masks showing different aspects of your personality or roles you play",
                'future_self': "Visualize and create artwork representing who you want to become"
            }
        }
        return descriptions.get(category, {}).get(exercise, "Therapeutic art exercise for self-expression")
    
    def _get_suitable_emotions(self, category, exercise):
        emotion_mapping = {
            'mindfulness': ['anxiety', 'stress', 'calm', 'neutral', 'confusion'],
            'emotional_expression': ['joy', 'sadness', 'anger', 'fear', 'surprise'],
            'trauma_recovery': ['fear', 'anxiety', 'calm', 'hope', 'despair'],
            'self_discovery': ['confusion', 'hope', 'neutral', 'calm', 'joy']
        }
        return emotion_mapping.get(category, [])
    
    def _get_required_materials(self, category):
        materials = {
            'mindfulness': ['paper', 'drawing materials', 'colored pencils', 'pens'],
            'emotional_expression': ['paint', 'brushes', 'colorful materials', 'large paper'],
            'trauma_recovery': ['clay or modeling materials', 'collage materials', 'symbolic objects'],
            'self_discovery': ['journal', 'mixed media', 'photographs', 'personal items']
        }
        return materials.get(category, ['basic art supplies'])
    
    def _fit_vectorizer(self):
        texts = self.exercise_database['description'].tolist()
        self.vectorizer.fit(texts)
    
    def recommend_exercises(self, user_emotion, art_style, user_experience='beginner', max_recommendations=5):
        user_profile = f"{user_emotion} {art_style} {user_experience}"
        
        user_vector = self.vectorizer.transform([user_profile])
        exercise_vectors = self.vectorizer.transform(self.exercise_database['description'])
        
        similarities = cosine_similarity(user_vector, exercise_vectors).flatten()
        
        self.exercise_database['similarity_score'] = similarities
        
        filtered_exercises = self.exercise_database[
            (self.exercise_database['difficulty'] == user_experience) |
            (user_emotion in self.exercise_database['suitable_emotions'])
        ]
        
        if len(filtered_exercises) == 0:
            filtered_exercises = self.exercise_database
        
        recommendations = filtered_exercises.nlargest(max_recommendations, 'similarity_score')
        
        return [
            {
                'category': row['category'],
                'exercise': row['exercise'],
                'description': row['description'],
                'duration': row['duration'],
                'materials': row['materials'],
                'suitability_score': float(row['similarity_score']),
                'ai_generated_prompt': self._generate_exercise_prompt(row, user_emotion, art_style)
            }
            for _, row in recommendations.iterrows()
        ]
    
    def _generate_exercise_prompt(self, exercise, user_emotion, art_style):
        prompt = f"""
        Create a detailed, step-by-step instruction for the art therapy exercise: {exercise['exercise']}
        Category: {exercise['category']}
        For someone feeling: {user_emotion}
        Preferred art style: {art_style}
        
        Make the instructions warm, supportive, and clinically appropriate. Include:
        1. A welcoming introduction
        2. Clear step-by-step instructions
        3. Supportive guidance for working with {user_emotion}
        4. Suggestions for reflection after the exercise
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compassionate art therapist. Provide clear, supportive instructions for therapeutic art exercises."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Begin by gathering your materials. Create artwork that expresses your experience of {user_emotion} using {art_style} techniques. Reflect on your creation afterward."
    
    def generate_progress_plan(self, user_goals, current_emotion, timeline_weeks=8):
        plan = {
            'timeline_weeks': timeline_weeks,
            'weekly_sessions': [],
            'goals': user_goals,
            'assessment_points': []
        }
        
        for week in range(1, timeline_weeks + 1):
            focus_area = self._get_weekly_focus(week, user_goals)
            session = {
                'week': week,
                'focus': focus_area,
                'recommended_exercises': self.recommend_exercises(current_emotion, 'varied', 'intermediate', 2),
                'reflection_questions': self._generate_reflection_questions(focus_area)
            }
            plan['weekly_sessions'].append(session)
            
            if week % 2 == 0:
                plan['assessment_points'].append(f"Week {week}: Review progress on {focus_area}")
        
        return plan
    
    def _get_weekly_focus(self, week, user_goals):
        focus_areas = {
            1: "Establishing safety and self-awareness",
            2: "Emotional identification and expression",
            3: "Exploring personal narratives",
            4: "Developing coping strategies",
            5: "Building resilience and strengths",
            6: "Integration and meaning-making",
            7: "Future orientation and goal-setting",
            8: "Consolidation and maintenance"
        }
        return focus_areas.get(week, "Therapeutic self-expression")
    
    def _generate_reflection_questions(self, focus_area):
        questions = {
            "Establishing safety and self-awareness": [
                "How did you feel during this art-making process?",
                "What colors or shapes felt most comfortable to work with?",
                "What did you discover about your current emotional state?"
            ],
            "Emotional identification and expression": [
                "What emotions emerged while creating this artwork?",
                "How did using art materials help express these feelings?",
                "What would you like to understand better about these emotions?"
            ],
            "Exploring personal narratives": [
                "What story does this artwork tell?",
                "How does this connect to your life experiences?",
                "What new perspectives emerged through this creation?"
            ]
        }
        return questions.get(focus_area, [
            "What was this art-making experience like for you?",
            "What insights emerged through the process?",
            "How might you apply this learning in your daily life?"
        ])