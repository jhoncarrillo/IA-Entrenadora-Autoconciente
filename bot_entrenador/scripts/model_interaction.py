from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import json

class AdvancedModelInteraction:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.interaction_history = []
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar modelos por defecto
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Inicializa modelos por defecto para diferentes tareas"""
        default_models = {
            'text_generation': 'gpt2',
            'text_classification': 'distilbert-base-uncased-finetuned-sst-2-english',
            'question_answering': 'distilbert-base-uncased-distilled-squad',
            'summarization': 'facebook/bart-large-cnn',
            'translation': 't5-small'
        }
        
        for task, model_name in default_models.items():
            try:
                self.load_model(task, model_name)
            except Exception as e:
                self.logger.warning(f"No se pudo cargar el modelo {model_name} para {task}: {str(e)}")
    
    def load_model(self, task_name: str, model_name: str, custom_config: Optional[Dict] = None):
        """Carga un modelo específico para una tarea"""
        try:
            if task_name == 'text_generation':
                self.tokenizers[task_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[task_name] = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Agregar pad_token si no existe
                if self.tokenizers[task_name].pad_token is None:
                    self.tokenizers[task_name].pad_token = self.tokenizers[task_name].eos_token
                    
            elif task_name == 'text_classification':
                self.pipelines[task_name] = pipeline('text-classification', model=model_name)
                
            elif task_name == 'question_answering':
                self.pipelines[task_name] = pipeline('question-answering', model=model_name)
                
            elif task_name == 'summarization':
                self.pipelines[task_name] = pipeline('summarization', model=model_name)
                
            elif task_name == 'translation':
                self.pipelines[task_name] = pipeline('translation', model=model_name)
            
            self.logger.info(f"Modelo {model_name} cargado exitosamente para {task_name}")
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo {model_name}: {str(e)}")
            raise
    
    def interact_with_model(self, prompt: str, task: str = 'text_generation', **kwargs) -> str:
        """Interactúa con un modelo específico"""
        try:
            if task == 'text_generation':
                return self._generate_text(prompt, **kwargs)
            elif task == 'text_classification':
                return self._classify_text(prompt, **kwargs)
            elif task == 'question_answering':
                return self._answer_question(prompt, **kwargs)
            elif task == 'summarization':
                return self._summarize_text(prompt, **kwargs)
            elif task == 'translation':
                return self._translate_text(prompt, **kwargs)
            else:
                raise ValueError(f"Tarea no soportada: {task}")
                
        except Exception as e:
            self.logger.error(f"Error en interacción con modelo para {task}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _generate_text(self, prompt: str, max_length: int = 150, temperature: float = 0.7, 
                      num_return_sequences: int = 1, **kwargs) -> str:
        """Genera texto usando modelo de generación"""
        if 'text_generation' not in self.models:
            raise ValueError("Modelo de generación de texto no disponible")
        
        tokenizer = self.tokenizers['text_generation']
        model = self.models['text_generation']
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Registrar interacción
        self._log_interaction('text_generation', prompt, response)
        
        return response
    
    def _classify_text(self, text: str, **kwargs) -> str:
        """Clasifica texto usando modelo de clasificación"""
        if 'text_classification' not in self.pipelines:
            raise ValueError("Modelo de clasificación no disponible")
        
        result = self.pipelines['text_classification'](text, **kwargs)
        
        if isinstance(result, list):
            result = result[0]
        
        response = f"Clasificación: {result['label']} (confianza: {result['score']:.3f})"
        self._log_interaction('text_classification', text, response)
        
        return response
    
    def _answer_question(self, question: str, context: str = "", **kwargs) -> str:
        """Responde preguntas usando modelo QA"""
        if 'question_answering' not in self.pipelines:
            raise ValueError("Modelo de QA no disponible")
        
        if not context:
            context = "No se proporcionó contexto específico."
        
        result = self.pipelines['question_answering'](question=question, context=context, **kwargs)
        
        response = f"Respuesta: {result['answer']} (confianza: {result['score']:.3f})"
        self._log_interaction('question_answering', f"Q: {question} | Context: {context[:100]}...", response)
        
        return response
    
    def _summarize_text(self, text: str, max_length: int = 150, min_length: int = 30, **kwargs) -> str:
        """Resume texto usando modelo de resumen"""
        if 'summarization' not in self.pipelines:
            raise ValueError("Modelo de resumen no disponible")
        
        result = self.pipelines['summarization'](
            text, 
            max_length=max_length, 
            min_length=min_length, 
            **kwargs
        )
        
        if isinstance(result, list):
            result = result[0]
        
        response = result['summary_text']
        self._log_interaction('summarization', text[:100] + "...", response)
        
        return response
    
    def _translate_text(self, text: str, target_language: str = "es", **kwargs) -> str:
        """Traduce texto usando modelo de traducción"""
        if 'translation' not in self.pipelines:
            raise ValueError("Modelo de traducción no disponible")
        
        # Para T5, necesitamos un prefijo específico
        if target_language == "es":
            text = f"translate English to Spanish: {text}"
        elif target_language == "fr":
            text = f"translate English to French: {text}"
        
        result = self.pipelines['translation'](text, **kwargs)
        
        if isinstance(result, list):
            result = result[0]
        
        response = result.get('translation_text', result.get('generated_text', str(result)))
        self._log_interaction('translation', text, response)
        
        return response
    
    def _log_interaction(self, task: str, input_text: str, output_text: str):
        """Registra la interacción para análisis posterior"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'input': input_text[:500],  # Limitar longitud
            'output': output_text[:500],
            'input_length': len(input_text),
            'output_length': len(output_text)
        }
        
        self.interaction_history.append(interaction)
        
        # Mantener solo las últimas 1000 interacciones
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
    
    def batch_process(self, inputs: List[str], task: str = 'text_generation', **kwargs) -> List[str]:
        """Procesa múltiples inputs en lote"""
        results = []
        
        for i, input_text in enumerate(inputs):
            try:
                result = self.interact_with_model(input_text, task, **kwargs)
                results.append(result)
                self.logger.info(f"Procesado {i+1}/{len(inputs)}")
            except Exception as e:
                self.logger.error(f"Error procesando input {i+1}: {str(e)}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def multi_model_consensus(self, prompt: str, tasks: List[str], **kwargs) -> Dict[str, str]:
        """Obtiene respuestas de múltiples modelos para consenso"""
        results = {}
        
        for task in tasks:
            try:
                result = self.interact_with_model(prompt, task, **kwargs)
                results[task] = result
            except Exception as e:
                results[task] = f"Error: {str(e)}"
        
        return results
    
    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analiza patrones en el historial de interacciones"""
        if not self.interaction_history:
            return {"message": "No hay historial de interacciones"}
        
        # Análisis básico
        task_counts = {}
        avg_input_length = 0
        avg_output_length = 0
        
        for interaction in self.interaction_history:
            task = interaction['task']
            task_counts[task] = task_counts.get(task, 0) + 1
            avg_input_length += interaction['input_length']
            avg_output_length += interaction['output_length']
        
        total_interactions = len(self.interaction_history)
        avg_input_length /= total_interactions
        avg_output_length /= total_interactions
        
        return {
            'total_interactions': total_interactions,
            'task_distribution': task_counts,
            'avg_input_length': avg_input_length,
            'avg_output_length': avg_output_length,
            'most_used_task': max(task_counts, key=task_counts.get) if task_counts else None,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def save_interaction_history(self, filepath: str):
        """Guarda el historial de interacciones"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.interaction_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Historial guardado en: {filepath}")
    
    def load_interaction_history(self, filepath: str):
        """Carga historial de interacciones desde archivo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.interaction_history = json.load(f)
            
            self.logger.info(f"Historial cargado desde: {filepath}")
        except Exception as e:
            self.logger.error(f"Error cargando historial: {str(e)}")
    
    def get_model_status(self) -> Dict[str, bool]:
        """Retorna el estado de todos los modelos cargados"""
        status = {}
        
        for task in ['text_generation', 'text_classification', 'question_answering', 'summarization', 'translation']:
            status[task] = (task in self.models) or (task in self.pipelines)
        
        return status

# Función de compatibilidad con el código existente
def interact_with_model(prompt: str, task: str = 'text_generation') -> str:
    """Función de compatibilidad para el código existente"""
    interaction_system = AdvancedModelInteraction()
    return interaction_system.interact_with_model(prompt, task)
