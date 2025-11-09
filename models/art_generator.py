# models/art_generator.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import openai
from config.settings import config

class ArtGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            config.GENERATIVE_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(self.device)
        
        openai.api_key = config.OPENAI_API_KEY
        
        self.therapeutic_prompts = self._load_therapeutic_prompts()
    
    def _load_therapeutic_prompts(self):
        return {
            'mindfulness': {
                'breathing_circle': "A serene, circular mandala with flowing patterns, soft blues and greens, representing calm breathing and mindfulness, therapeutic art, peaceful, meditative",
                'mandala': "An intricate, symmetrical mandala with harmonious colors, representing inner balance and wholeness, detailed patterns, spiritual, centering",
                'texture_exploration': "Abstract artwork exploring different textures and materials, tactile surfaces, mixed media, sensory exploration, therapeutic"
            },
            'emotional_expression': {
                'emotion_color_wheel': "A vibrant color wheel where each color represents a different emotion, expressive brushstrokes, emotional landscape, cathartic art",
                'feeling_landscape': "An abstract landscape that visualizes inner emotional states, metaphorical scenery, emotional weather, expressive colors",
                'inner_child': "Playful, naive art style representing the inner child, bright colors, simple shapes, joyful expression, therapeutic healing"
            },
            'trauma_recovery': {
                'safe_space': "A warm, secure, comforting environment visualized through art, protective atmosphere, healing space, sanctuary, therapeutic",
                'container_exercise': "A strong, beautiful container holding difficult emotions, symbolic representation, secure vessel, transformative art",
                'body_map': "An artistic representation of the body showing areas of tension and release, somatic awareness, healing visualization"
            },
            'self_discovery': {
                'life_tree': "A symbolic tree representing personal growth and life journey, roots and branches, seasonal changes, self-discovery art",
                'mask_exercise': "Artistic masks showing different aspects of personality, layers of self, authentic expression, identity exploration",
                'future_self': "A hopeful visualization of one's future self, aspirational art, positive projection, growth-oriented imagery"
            }
        }
    
    def generate_therapeutic_art(self, exercise_type, user_emotion=None, style_preference=None):
        if exercise_type not in self.therapeutic_prompts:
            exercise_type = 'mindfulness'
        
        exercise_options = self.therapeutic_prompts[exercise_type]
        specific_exercise = list(exercise_options.keys())[0]
        
        base_prompt = exercise_options[specific_exercise]
        
        if user_emotion:
            emotion_modifier = self._get_emotion_modifier(user_emotion)
            prompt = f"{base_prompt}, {emotion_modifier}"
        else:
            prompt = base_prompt
        
        if style_preference:
            prompt = f"{prompt}, in {style_preference} style"
        
        negative_prompt = "violent, disturbing, scary, ugly, deformed, low quality, blurry"
        
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512
        ).images[0]
        
        return {
            'image': image,
            'prompt_used': prompt,
            'exercise_type': exercise_type,
            'specific_exercise': specific_exercise,
            'therapeutic_intent': self._get_therapeutic_intent(exercise_type, specific_exercise)
        }
    
    def _get_emotion_modifier(self, emotion):
        modifiers = {
            'joy': "vibrant, uplifting, celebratory, warm colors",
            'sadness': "gentle, reflective, soft transitions, cool tones",
            'anger': "dynamic, intense, bold strokes, high contrast",
            'fear': "protective, contained, soft edges, reassuring colors",
            'calm': "serene, balanced, flowing, harmonious palette",
            'anxiety': "grounding, structured, repetitive patterns, stable"
        }
        return modifiers.get(emotion, "expressive, meaningful, therapeutic")
    
    def _get_therapeutic_intent(self, category, exercise):
        intents = {
            'mindfulness': {
                'breathing_circle': "Focus on breath and present moment awareness",
                'mandala': "Promote concentration and inner peace through pattern creation",
                'texture_exploration': "Enhance sensory awareness and grounding"
            },
            'emotional_expression': {
                'emotion_color_wheel': "Identify and express complex emotional states",
                'feeling_landscape': "Externalize internal emotional experiences",
                'inner_child': "Connect with and nurture playful, authentic self"
            },
            'trauma_recovery': {
                'safe_space': "Create internal sense of safety and security",
                'container_exercise': "Develop capacity to contain difficult emotions",
                'body_map': "Increase body awareness and release somatic tension"
            },
            'self_discovery': {
                'life_tree': "Explore personal history and growth potential",
                'mask_exercise': "Examine different aspects of identity and self-presentation",
                'future_self': "Visualize and work toward positive future outcomes"
            }
        }
        return intents.get(category, {}).get(exercise, "Promote self-expression and insight")
    
    def generate_personalized_prompt(self, user_emotion, art_style, therapeutic_goal):
        prompt = f"""
        Create a therapeutic art prompt for someone feeling {user_emotion} 
        who wants to work on {therapeutic_goal} through {art_style} style artwork.
        The prompt should be encouraging, therapeutic, and specific enough to guide artistic expression.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an experienced art therapist. Create therapeutic art prompts that are supportive, specific, and clinically appropriate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Create {art_style} artwork expressing {user_emotion} while focusing on {therapeutic_goal}. Use colors and forms that feel authentic to your experience."