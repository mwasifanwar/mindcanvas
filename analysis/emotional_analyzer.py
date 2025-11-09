# analysis/emotional_analyzer.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionalAnalyzer:
    def __init__(self):
        self.emotion_categories = {
            'positive': ['joy', 'calm', 'hope', 'surprise'],
            'negative': ['sadness', 'anger', 'fear', 'anxiety', 'despair'],
            'neutral': ['neutral', 'confusion']
        }
    
    def analyze_emotional_trends(self, session_data):
        df = pd.DataFrame(session_data)
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        
        weekly_trends = df.resample('W').agg({
            'primary_emotion': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral',
            'intensity': 'mean',
            'emotional_complexity': 'mean'
        })
        
        emotion_transitions = self._calculate_emotion_transitions(df)
        stability_score = self._calculate_emotional_stability(df)
        
        return {
            'weekly_trends': weekly_trends.to_dict('records'),
            'dominant_emotions': self._get_dominant_emotions(df),
            'emotion_transitions': emotion_transitions,
            'stability_score': stability_score,
            'growth_indicators': self._assess_growth_indicators(df)
        }
    
    def _calculate_emotion_transitions(self, df):
        transitions = []
        emotions = df['primary_emotion'].tolist()
        
        for i in range(1, len(emotions)):
            from_emotion = emotions[i-1]
            to_emotion = emotions[i]
            
            transition_type = self._classify_transition(from_emotion, to_emotion)
            transitions.append({
                'from': from_emotion,
                'to': to_emotion,
                'type': transition_type,
                'session_gap': i
            })
        
        return transitions
    
    def _classify_transition(self, from_emotion, to_emotion):
        from_category = self._categorize_emotion(from_emotion)
        to_category = self._categorize_emotion(to_emotion)
        
        if from_category == to_category:
            return 'stable'
        elif from_category == 'negative' and to_category == 'positive':
            return 'improving'
        elif from_category == 'positive' and to_category == 'negative':
            return 'declining'
        else:
            return 'mixed'
    
    def _categorize_emotion(self, emotion):
        for category, emotions in self.emotion_categories.items():
            if emotion in emotions:
                return category
        return 'neutral'
    
    def _calculate_emotional_stability(self, df):
        emotion_changes = 0
        emotions = df['primary_emotion'].tolist()
        
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                emotion_changes += 1
        
        if len(emotions) <= 1:
            return 1.0
        
        stability = 1 - (emotion_changes / (len(emotions) - 1))
        return stability
    
    def _get_dominant_emotions(self, df):
        emotion_counts = df['primary_emotion'].value_counts()
        return emotion_counts.to_dict()
    
    def _assess_growth_indicators(self, df):
        indicators = {}
        
        if len(df) >= 3:
            recent_intensity = df['intensity'].tail(3).mean()
            earlier_intensity = df['intensity'].head(3).mean()
            indicators['intensity_trend'] = 'decreasing' if recent_intensity < earlier_intensity else 'increasing'
            
            recent_complexity = df['emotional_complexity'].tail(3).mean()
            earlier_complexity = df['emotional_complexity'].head(3).mean()
            indicators['complexity_trend'] = 'increasing' if recent_complexity > earlier_complexity else 'decreasing'
        
        positive_ratio = len(df[df['primary_emotion'].isin(self.emotion_categories['positive'])]) / len(df)
        indicators['positive_expression_ratio'] = positive_ratio
        
        return indicators
    
    def generate_emotional_insights(self, trend_analysis):
        insights = []
        
        stability = trend_analysis['stability_score']
        if stability > 0.7:
            insights.append("Good emotional stability observed across sessions")
        elif stability < 0.3:
            insights.append("Significant emotional variability - may indicate processing of complex feelings")
        
        positive_ratio = trend_analysis['growth_indicators'].get('positive_expression_ratio', 0)
        if positive_ratio > 0.6:
            insights.append("Strong positive emotional expression in artwork")
        elif positive_ratio < 0.2:
            insights.append("Limited positive emotional expression - consider focusing on strengths and resources")
        
        complexity_trend = trend_analysis['growth_indicators'].get('complexity_trend')
        if complexity_trend == 'increasing':
            insights.append("Growing emotional complexity suggests deepening self-awareness")
        
        return insights
    
    def create_emotional_timeline_visualization(self, session_data):
        df = pd.DataFrame(session_data)
        df['date'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 8))
        
        emotion_colors = {
            'joy': 'gold', 'sadness': 'blue', 'anger': 'red', 'fear': 'purple',
            'calm': 'green', 'anxiety': 'orange', 'neutral': 'gray', 'hope': 'pink'
        }
        
        for emotion, color in emotion_colors.items():
            emotion_data = df[df['primary_emotion'] == emotion]
            if not emotion_data.empty:
                plt.scatter(emotion_data['date'], emotion_data['intensity'], 
                           c=color, label=emotion, alpha=0.7, s=100)
        
        plt.plot(df['date'], df['intensity'], 'k-', alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Emotional Intensity')
        plt.title('Emotional Timeline')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()