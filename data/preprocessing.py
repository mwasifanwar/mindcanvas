# data/preprocessing.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from scipy import ndimage

class ArtPreprocessor:
    def __init__(self):
        self.color_ranges = {
            'warm': [(0, 50, 50), (30, 255, 255)],
            'cool': [(100, 50, 50), (140, 255, 255)],
            'neutral': [(0, 0, 40), (180, 50, 255)]
        }
    
    def analyze_color_palette(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        image_np = np.array(image)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        color_distribution = {}
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_distribution[color_name] = np.sum(mask) / (mask.size + 1e-6)
        
        dominant_color = max(color_distribution, key=color_distribution.get)
        
        return {
            'color_distribution': color_distribution,
            'dominant_color': dominant_color,
            'color_variance': np.var(list(color_distribution.values()))
        }
    
    def extract_composition_features(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.size * 255)
        
        blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        brightness = np.mean(gray) / 255
        contrast = gray.std() / 255
        
        return {
            'edge_density': edge_density,
            'blur_variance': blur_variance,
            'brightness': brightness,
            'contrast': contrast,
            'symmetry_score': self._calculate_symmetry(gray)
        }
    
    def _calculate_symmetry(self, gray_image):
        height, width = gray_image.shape
        vertical_flip = cv2.flip(gray_image, 1)
        horizontal_flip = cv2.flip(gray_image, 0)
        
        vertical_symmetry = np.corrcoef(gray_image.flatten(), vertical_flip.flatten())[0,1]
        horizontal_symmetry = np.corrcoef(gray_image.flatten(), horizontal_flip.flatten())[0,1]
        
        return (vertical_symmetry + horizontal_symmetry) / 2
    
    def detect_emotional_cues(self, image):
        color_analysis = self.analyze_color_palette(image)
        composition = self.extract_composition_features(image)
        
        emotional_score = self._compute_emotional_score(color_analysis, composition)
        
        return {
            'color_analysis': color_analysis,
            'composition_analysis': composition,
            'emotional_score': emotional_score,
            'predicted_emotion': self._map_to_emotion(emotional_score)
        }
    
    def _compute_emotional_score(self, color_analysis, composition):
        warm_ratio = color_analysis['color_distribution']['warm']
        cool_ratio = color_analysis['color_distribution']['cool']
        brightness = composition['brightness']
        contrast = composition['contrast']
        edge_density = composition['edge_density']
        
        emotional_intensity = (warm_ratio * 0.3 + cool_ratio * 0.2 + 
                             brightness * 0.2 + contrast * 0.15 + edge_density * 0.15)
        
        emotional_valence = (warm_ratio - cool_ratio) * 0.5 + brightness * 0.3 + contrast * 0.2
        
        return {
            'intensity': emotional_intensity,
            'valence': emotional_valence,
            'arousal': edge_density * 0.6 + contrast * 0.4
        }
    
    def _map_to_emotion(self, emotional_score):
        valence = emotional_score['valence']
        arousal = emotional_score['arousal']
        
        if valence > 0.6 and arousal > 0.6:
            return 'joy'
        elif valence > 0.6 and arousal <= 0.6:
            return 'calm'
        elif valence <= 0.6 and arousal > 0.6:
            return 'anger'
        elif valence <= 0.3 and arousal <= 0.3:
            return 'sadness'
        elif arousal > 0.7:
            return 'fear'
        else:
            return 'neutral'
    
    def enhance_therapeutic_qualities(self, image, target_emotion='calm'):
        if isinstance(image, str):
            image = Image.open(image)
        
        enhancers = {
            'calm': self._enhance_calmness,
            'joy': self._enhance_joy,
            'sadness': self._enhance_sadness,
            'anger': self._enhance_anger
        }
        
        enhancer = enhancers.get(target_emotion, self._enhance_calmness)
        return enhancer(image)
    
    def _enhance_calmness(self, image):
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.9)
        return image.filter(ImageFilter.SMOOTH)
    
    def _enhance_joy(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.3)
        return image
    
    def _enhance_sadness(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.7)
        return image
    
    def _enhance_anger(self, image):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        return image