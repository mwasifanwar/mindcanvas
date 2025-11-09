# models/emotion_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
import numpy as np

class MultiModalEmotionClassifier(nn.Module):
    def __init__(self, num_emotions=12, num_styles=8, hidden_dim=512):
        super(MultiModalEmotionClassifier, self).__init__()
        
        self.vision_encoder = models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Linear(self.vision_encoder.fc.in_features, hidden_dim)
        
        self.text_encoder = AutoModel.from_pretrained(config.NLP_MODEL_NAME)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.emotion_classifier = nn.Linear(hidden_dim // 2, num_emotions)
        self.style_classifier = nn.Linear(hidden_dim // 2, num_styles)
        self.intensity_regressor = nn.Linear(hidden_dim // 2, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, image, text_inputs=None):
        visual_features = self.vision_encoder(image)
        
        if text_inputs is not None:
            text_outputs = self.text_encoder(**text_inputs)
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_projection(text_features)
        else:
            text_features = torch.zeros_like(visual_features)
        
        fused_features = torch.cat([visual_features, text_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        emotion_logits = self.emotion_classifier(fused_features)
        style_logits = self.style_classifier(fused_features)
        intensity = self.sigmoid(self.intensity_regressor(fused_features))
        
        return {
            'emotions': self.softmax(emotion_logits),
            'styles': self.softmax(style_logits),
            'intensity': intensity.squeeze()
        }

class EmotionClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiModalEmotionClassifier()
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.NLP_MODEL_NAME)
        self.preprocessor = ArtPreprocessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def analyze_artwork(self, image_path, user_description=None):
        image_tensor = self._preprocess_image(image_path)
        
        text_inputs = None
        if user_description:
            text_inputs = self.tokenizer(
                user_description, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(image_tensor, text_inputs)
        
        cv_analysis = self.preprocessor.detect_emotional_cues(image_path)
        
        return self._format_analysis(outputs, cv_analysis)
    
    def _preprocess_image(self, image_path):
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(self.device)
    
    def _format_analysis(self, model_outputs, cv_analysis):
        emotion_probs = model_outputs['emotions'].cpu().numpy()[0]
        style_probs = model_outputs['styles'].cpu().numpy()[0]
        intensity = model_outputs['intensity'].cpu().numpy()
        
        top_emotions = sorted(
            zip(config.EMOTION_LABELS, emotion_probs),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        dominant_style = config.ART_STYLES[np.argmax(style_probs)]
        
        return {
            'emotional_analysis': {
                'primary_emotion': top_emotions[0][0],
                'primary_confidence': float(top_emotions[0][1]),
                'secondary_emotions': [
                    {'emotion': emo, 'confidence': float(conf)} 
                    for emo, conf in top_emotions[1:]
                ],
                'emotional_intensity': float(intensity),
                'emotional_complexity': len([p for p in emotion_probs if p > 0.1])
            },
            'artistic_analysis': {
                'dominant_style': dominant_style,
                'style_confidence': float(np.max(style_probs)),
                'color_analysis': cv_analysis['color_analysis'],
                'composition_analysis': cv_analysis['composition_analysis']
            },
            'therapeutic_insights': self._generate_therapeutic_insights(
                top_emotions[0][0], 
                cv_analysis['emotional_score']
            )
        }
    
    def _generate_therapeutic_insights(self, primary_emotion, emotional_score):
        insights = {
            'emotional_state': primary_emotion,
            'recommended_focus': '',
            'potential_concerns': [],
            'positive_aspects': []
        }
        
        focus_mapping = {
            'joy': 'exploration and celebration',
            'sadness': 'processing and release',
            'anger': 'channeling and transformation',
            'fear': 'safety and grounding',
            'calm': 'mindfulness and integration',
            'anxiety': 'containment and regulation'
        }
        
        insights['recommended_focus'] = focus_mapping.get(primary_emotion, 'self-expression')
        
        if emotional_score['intensity'] > 0.8:
            insights['potential_concerns'].append('High emotional intensity detected')
        if emotional_score['valence'] < 0.2:
            insights['potential_concerns'].append('Low positive emotional valence')
        if emotional_score['arousal'] > 0.7:
            insights['potential_concerns'].append('High arousal level - consider calming exercises')
        
        if emotional_score['valence'] > 0.6:
            insights['positive_aspects'].append('Strong positive emotional expression')
        if emotional_score['intensity'] < 0.4:
            insights['positive_aspects'].append('Balanced emotional expression')
        
        return insights