from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import json
from typing import Dict, List, Optional, Union

class AdvancedPromptGenerator:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Agregar pad_token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_prompt_for_transformers(self, question: str, context: Optional[str] = None) -> str:
        """Genera un prompt optimizado para modelos transformer"""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        return prompt
    
    def generate_training_prompts(self, data: Union[str, List[str], Dict]) -> List[str]:
        """Genera múltiples prompts para entrenamiento basados en los datos"""
        prompts = []
        
        if isinstance(data, str):
            prompts.extend([
                f"Analyze this text and extract key insights: {data[:500]}...",
                f"Summarize the main points from: {data[:500]}...",
                f"What patterns can be identified in: {data[:500]}...",
                f"Generate questions based on: {data[:500]}..."
            ])
        elif isinstance(data, list):
            for item in data[:5]:  # Limitar a 5 elementos
                prompts.append(f"Process this data point: {str(item)[:200]}...")
        elif isinstance(data, dict):
            for key, value in list(data.items())[:5]:
                prompts.append(f"Analyze the relationship between {key} and {str(value)[:200]}...")
        
        return prompts
    
    def create_few_shot_prompt(self, examples: List[Dict[str, str]], question: str) -> str:
        """Crea un prompt few-shot con ejemplos"""
        prompt = "Here are some examples:\n\n"
        
        for i, example in enumerate(examples[:3]):  # Máximo 3 ejemplos
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {example.get('input', '')}\n"
            prompt += f"Output: {example.get('output', '')}\n\n"
        
        prompt += f"Now, please process this:\nInput: {question}\nOutput:"
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Genera respuesta usando el modelo"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

# Función de compatibilidad con el código existente
def create_prompt_for_transformers(question: str, context: Optional[str] = None) -> str:
    """Función de compatibilidad para el código existente"""
    generator = AdvancedPromptGenerator()
    return generator.create_prompt_for_transformers(question, context)
