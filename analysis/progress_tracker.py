# analysis/progress_tracker.py
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class ProgressTracker:
    def __init__(self, storage_path="data/progress_data.json"):
        self.storage_path = storage_path
        self.data = self._load_data()
    
    def _load_data(self):
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except:
            return {'users': {}, 'sessions': []}
    
    def _save_data(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_session(self, user_id, art_analysis, exercise_completed, user_feedback=None):
        session_id = f"session_{len(self.data['sessions']) + 1}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'art_analysis': art_analysis,
            'exercise_completed': exercise_completed,
            'user_feedback': user_feedback,
            'progress_metrics': self._calculate_session_metrics(art_analysis)
        }
        
        self.data['sessions'].append(session_data)
        
        if user_id not in self.data['users']:
            self.data['users'][user_id] = {
                'created': datetime.now().isoformat(),
                'sessions': [],
                'goals': []
            }
        
        self.data['users'][user_id]['sessions'].append(session_id)
        self._save_data()
        
        return session_data
    
    def _calculate_session_metrics(self, art_analysis):
        emotional = art_analysis['emotional_analysis']
        artistic = art_analysis['artistic_analysis']
        
        engagement_score = (emotional['emotional_intensity'] + 
                          artistic['composition_analysis']['edge_density'] +
                          artistic['color_analysis']['color_variance']) / 3
        
        expressiveness_score = (emotional['emotional_complexity'] * 0.4 +
                              artistic['composition_analysis']['contrast'] * 0.3 +
                              emotional['primary_confidence'] * 0.3)
        
        return {
            'engagement_score': engagement_score,
            'expressiveness_score': expressiveness_score,
            'emotional_awareness': emotional['emotional_complexity'],
            'artistic_exploration': artistic['color_analysis']['color_variance']
        }
    
    def get_user_progress(self, user_id, timeframe_days=30):
        user_sessions = [s for s in self.data['sessions'] 
                        if s['user_id'] == user_id]
        
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        recent_sessions = [s for s in user_sessions 
                          if datetime.fromisoformat(s['timestamp']) > cutoff_date]
        
        if not recent_sessions:
            return None
        
        progress_metrics = []
        for session in recent_sessions:
            metrics = session['progress_metrics']
            metrics['date'] = session['timestamp']
            progress_metrics.append(metrics)
        
        df = pd.DataFrame(progress_metrics)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        progress_summary = {
            'total_sessions': len(recent_sessions),
            'time_period_days': timeframe_days,
            'average_engagement': df['engagement_score'].mean(),
            'average_expressiveness': df['expressiveness_score'].mean(),
            'trend_engagement': self._calculate_trend(df['engagement_score']),
            'trend_expressiveness': self._calculate_trend(df['expressiveness_score']),
            'recent_improvement': self._calculate_recent_improvement(df),
            'consistency_score': self._calculate_consistency(df)
        }
        
        return progress_summary
    
    def _calculate_trend(self, series):
        if len(series) < 2:
            return 'stable'
        
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_recent_improvement(self, df):
        if len(df) < 3:
            return 0
        
        recent_avg = df.tail(3)['engagement_score'].mean()
        earlier_avg = df.head(3)['engagement_score'].mean()
        
        return (recent_avg - earlier_avg) / (earlier_avg + 1e-6)
    
    def _calculate_consistency(self, df):
        if len(df) < 2:
            return 1.0
        
        engagement_std = df['engagement_score'].std()
        engagement_mean = df['engagement_score'].mean()
        
        if engagement_mean == 0:
            return 1.0
        
        consistency = 1 - (engagement_std / engagement_mean)
        return max(0, min(1, consistency))
    
    def generate_progress_report(self, user_id):
        progress = self.get_user_progress(user_id)
        
        if not progress:
            return {"error": "No progress data available"}
        
        report = {
            'user_id': user_id,
            'report_date': datetime.now().isoformat(),
            'summary': progress,
            'recommendations': self._generate_recommendations(progress),
            'milestones': self._identify_milestones(user_id),
            'next_steps': self._suggest_next_steps(progress)
        }
        
        return report
    
    def _generate_recommendations(self, progress):
        recommendations = []
        
        if progress['trend_engagement'] == 'declining':
            recommendations.append("Consider trying different art materials or exercises to renew engagement")
        
        if progress['average_expressiveness'] < 0.4:
            recommendations.append("Focus on emotional expression exercises to enhance artistic expressiveness")
        
        if progress['consistency_score'] < 0.6:
            recommendations.append("Establish a regular art-making routine to build consistency")
        
        if progress['recent_improvement'] > 0.1:
            recommendations.append("Continue with current approach - showing positive progress")
        
        if not recommendations:
            recommendations.append("Maintain current practice and explore new creative directions")
        
        return recommendations
    
    def _identify_milestones(self, user_id):
        user_sessions = [s for s in self.data['sessions'] 
                        if s['user_id'] == user_id]
        
        milestones = []
        
        if len(user_sessions) >= 5:
            milestones.append("Completed 5 art therapy sessions")
        
        if len(user_sessions) >= 10:
            milestones.append("Completed 10 sessions - building consistent practice")
        
        engagement_scores = [s['progress_metrics']['engagement_score'] 
                           for s in user_sessions[-3:]]
        if len(engagement_scores) >= 3 and min(engagement_scores) > 0.7:
            milestones.append("Sustained high engagement in recent sessions")
        
        return milestones
    
    def _suggest_next_steps(self, progress):
        next_steps = []
        
        if progress['total_sessions'] < 5:
            next_steps.append("Continue exploring different art materials and techniques")
        elif progress['total_sessions'] < 15:
            next_steps.append("Begin working on more complex emotional themes")
        else:
            next_steps.append("Consider deepening specific therapeutic goals")
        
        if progress['average_expressiveness'] < 0.5:
            next_steps.append("Practice free-form expressive art exercises")
        
        return next_steps