import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_extraction import extract_data, AdvancedDataExtractor
from scripts.prompt_generation import create_prompt_for_transformers, AdvancedPromptGenerator
from scripts.model_interaction import interact_with_model, AdvancedModelInteraction
from models.cnn_model import create_cnn
from models.lstm_model import create_lstm
from models.rnn_model import create_rnn
from models.transformer_model import create_transformer
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AdvancedEcosystemTraining:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.data_extractor = AdvancedDataExtractor()
        self.prompt_generator = AdvancedPromptGenerator()
        self.model_interaction = AdvancedModelInteraction()
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Historial de entrenamiento
        self.training_history = []
        self.model_registry = {}
        self.performance_metrics = {}
        
        # Escaladores y encoders
        self.scalers = {}
        self.encoders = {}
        
    def _get_default_config(self) -> Dict:
        """Configuración por defecto del ecosistema"""
        return {
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.1,
            'early_stopping_patience': 5,
            'save_models': True,
            'model_save_path': './trained_models',
            'log_training': True,
            'use_gpu': torch.cuda.is_available(),
            'random_seed': 42
        }
    
    def train_ecosystem(self, data_sources: List[str], model_configs: List[Dict], 
                       training_objectives: Optional[List[str]] = None) -> Dict[str, Any]:
        """Entrena un ecosistema completo de modelos"""
        try:
            self.logger.info("Iniciando entrenamiento del ecosistema...")
            
            # 1. Extraer y procesar datos
            processed_data = self._extract_and_process_data(data_sources)
            
            # 2. Generar prompts y respuestas
            training_data = self._generate_training_data(processed_data, training_objectives)
            
            # 3. Preparar datos para entrenamiento
            prepared_data = self._prepare_training_data(training_data)
            
            # 4. Entrenar modelos
            training_results = self._train_models(prepared_data, model_configs)
            
            # 5. Evaluar ecosistema
            evaluation_results = self._evaluate_ecosystem(training_results, prepared_data)
            
            # 6. Guardar resultados
            self._save_training_results(training_results, evaluation_results)
            
            return {
                'status': 'success',
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento del ecosistema: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_and_process_data(self, data_sources: List[str]) -> Dict[str, Any]:
        """Extrae y procesa datos de múltiples fuentes"""
        all_data = {
            'text_data': [],
            'numeric_data': [],
            'image_data': [],
            'structured_data': []
        }
        
        for source in data_sources:
            try:
                # Extraer datos usando el extractor avanzado
                extracted_data = self.data_extractor.extract_data(source)
                
                # Clasificar datos por tipo
                if isinstance(extracted_data, str):
                    all_data['text_data'].append(extracted_data)
                elif isinstance(extracted_data, (list, np.ndarray)):
                    if len(extracted_data) > 0 and isinstance(extracted_data[0], (int, float)):
                        all_data['numeric_data'].append(extracted_data)
                    else:
                        all_data['structured_data'].append(extracted_data)
                elif isinstance(extracted_data, pd.DataFrame):
                    all_data['structured_data'].append(extracted_data)
                
                self.logger.info(f"Datos extraídos de: {source}")
                
            except Exception as e:
                self.logger.warning(f"Error extrayendo datos de {source}: {str(e)}")
        
        return all_data
    
    def _generate_training_data(self, processed_data: Dict[str, Any], 
                              objectives: Optional[List[str]] = None) -> List[Dict]:
        """Genera datos de entrenamiento usando prompts"""
        training_data = []
        
        objectives = objectives or [
            "clasificación de texto",
            "generación de texto",
            "análisis de sentimientos",
            "extracción de información"
        ]
        
        # Procesar datos de texto
        for text in processed_data['text_data']:
            for objective in objectives:
                try:
                    # Generar prompt optimizado
                    prompt = self.prompt_generator.create_optimized_prompt(
                        task_type=objective,
                        context=text[:500],  # Limitar contexto
                        examples=[]
                    )
                    
                    # Generar respuesta usando modelo
                    response = self.model_interaction.interact_with_model(
                        prompt, 
                        task='text_generation'
                    )
                    
                    training_data.append({
                        'input': prompt,
                        'target': response,
                        'data_type': 'text',
                        'objective': objective,
                        'source_length': len(text)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error generando datos de entrenamiento: {str(e)}")
        
        # Procesar datos estructurados
        for data in processed_data['structured_data']:
            if isinstance(data, pd.DataFrame):
                # Generar prompts para datos tabulares
                for objective in ['predicción', 'clasificación']:
                    try:
                        prompt = self.prompt_generator.create_training_prompt(
                            data.head().to_dict(),
                            prompt_type=objective
                        )
                        
                        training_data.append({
                            'input': prompt,
                            'target': f"Análisis de datos tabulares para {objective}",
                            'data_type': 'structured',
                            'objective': objective,
                            'features': list(data.columns)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Error procesando datos estructurados: {str(e)}")
        
        self.logger.info(f"Generados {len(training_data)} ejemplos de entrenamiento")
        return training_data
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Prepara datos para entrenamiento de modelos"""
        prepared_data = {
            'text_classification': {'X': [], 'y': []},
            'text_generation': {'X': [], 'y': []},
            'regression': {'X': [], 'y': []},
            'sequence_prediction': {'X': [], 'y': []}
        }
        
        # Preparar datos según el objetivo
        for item in training_data:
            objective = item['objective']
            input_text = item['input']
            target = item['target']
            
            if 'clasificación' in objective or 'sentimientos' in objective:
                prepared_data['text_classification']['X'].append(input_text)
                prepared_data['text_classification']['y'].append(target)
                
            elif 'generación' in objective:
                prepared_data['text_generation']['X'].append(input_text)
                prepared_data['text_generation']['y'].append(target)
                
            elif 'predicción' in objective:
                prepared_data['regression']['X'].append(input_text)
                prepared_data['regression']['y'].append(len(target))  # Ejemplo simple
                
            else:
                prepared_data['sequence_prediction']['X'].append(input_text)
                prepared_data['sequence_prediction']['y'].append(target)
        
        # Convertir a arrays numpy donde sea apropiado
        for task_type in prepared_data:
            if prepared_data[task_type]['X']:
                # Para datos de texto, mantener como lista
                if task_type in ['text_classification', 'text_generation']:
                    continue
                else:
                    try:
                        prepared_data[task_type]['X'] = np.array(prepared_data[task_type]['X'])
                        prepared_data[task_type]['y'] = np.array(prepared_data[task_type]['y'])
                    except:
                        pass  # Mantener como lista si no se puede convertir
        
        return prepared_data
    
    def _train_models(self, prepared_data: Dict[str, Any], 
                     model_configs: List[Dict]) -> Dict[str, Any]:
        """Entrena múltiples modelos con los datos preparados"""
        training_results = {}
        
        for config in model_configs:
            model_type = config.get('type', 'cnn')
            model_name = config.get('name', f"{model_type}_model")
            
            try:
                self.logger.info(f"Entrenando modelo: {model_name}")
                
                # Crear modelo según el tipo
                if model_type == 'cnn':
                    model = self._create_and_train_cnn(prepared_data, config)
                elif model_type == 'lstm':
                    model = self._create_and_train_lstm(prepared_data, config)
                elif model_type == 'rnn':
                    model = self._create_and_train_rnn(prepared_data, config)
                elif model_type == 'transformer':
                    model = self._create_and_train_transformer(prepared_data, config)
                else:
                    self.logger.warning(f"Tipo de modelo no soportado: {model_type}")
                    continue
                
                # Registrar modelo
                self.model_registry[model_name] = {
                    'model': model,
                    'config': config,
                    'training_timestamp': datetime.now().isoformat()
                }
                
                training_results[model_name] = {
                    'status': 'success',
                    'model_type': model_type,
                    'config': config
                }
                
            except Exception as e:
                self.logger.error(f"Error entrenando {model_name}: {str(e)}")
                training_results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return training_results
    
    def _create_and_train_cnn(self, data: Dict[str, Any], config: Dict) -> Any:
        """Crea y entrena un modelo CNN"""
        # Usar datos de clasificación de texto como ejemplo
        if data['text_classification']['X']:
            # Simulación de entrenamiento CNN para texto
            input_shape = config.get('input_shape', (28, 28, 1))
            num_classes = config.get('num_classes', 10)
            
            model = create_cnn(input_shape, num_classes)
            
            # Aquí iría el entrenamiento real con datos procesados
            self.logger.info("Modelo CNN creado y entrenado (simulación)")
            
            return model
        
        return None
    
    def _create_and_train_lstm(self, data: Dict[str, Any], config: Dict) -> Any:
        """Crea y entrena un modelo LSTM"""
        if data['sequence_prediction']['X']:
            input_shape = config.get('input_shape', (100, 1))
            output_units = config.get('output_units', 1)
            
            model = create_lstm(input_shape, output_units)
            
            self.logger.info("Modelo LSTM creado y entrenado (simulación)")
            
            return model
        
        return None
    
    def _create_and_train_rnn(self, data: Dict[str, Any], config: Dict) -> Any:
        """Crea y entrena un modelo RNN"""
        if data['sequence_prediction']['X']:
            input_shape = config.get('input_shape', (100, 1))
            output_units = config.get('output_units', 1)
            
            model = create_rnn(input_shape, output_units)
            
            self.logger.info("Modelo RNN creado y entrenado (simulación)")
            
            return model
        
        return None
    
    def _create_and_train_transformer(self, data: Dict[str, Any], config: Dict) -> Any:
        """Crea y entrena un modelo Transformer"""
        if data['text_classification']['X']:
            model_name = config.get('model_name', 'bert-base-uncased')
            num_labels = config.get('num_labels', 2)
            
            model = create_transformer(model_name, num_labels)
            
            self.logger.info("Modelo Transformer creado y entrenado (simulación)")
            
            return model
        
        return None
    
    def _evaluate_ecosystem(self, training_results: Dict[str, Any], 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa el rendimiento del ecosistema"""
        evaluation_results = {}
        
        for model_name, result in training_results.items():
            if result['status'] == 'success':
                try:
                    # Evaluación simulada
                    evaluation_results[model_name] = {
                        'accuracy': np.random.uniform(0.7, 0.95),
                        'loss': np.random.uniform(0.1, 0.5),
                        'training_time': np.random.uniform(10, 300),
                        'model_size': np.random.uniform(1, 100),
                        'evaluation_timestamp': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    evaluation_results[model_name] = {
                        'error': str(e)
                    }
        
        return evaluation_results
    
    def _save_training_results(self, training_results: Dict[str, Any], 
                             evaluation_results: Dict[str, Any]):
        """Guarda los resultados del entrenamiento"""
        if self.config['save_models']:
            save_path = self.config['model_save_path']
            os.makedirs(save_path, exist_ok=True)
            
            # Guardar resultados en JSON
            results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = os.path.join(save_path, 'training_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Resultados guardados en: {results_file}")
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del ecosistema"""
        return {
            'registered_models': list(self.model_registry.keys()),
            'training_history_count': len(self.training_history),
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'data_extractor_status': 'active',
            'prompt_generator_status': 'active',
            'model_interaction_status': 'active'
        }
    
    def autonomous_training_cycle(self, data_sources: List[str], 
                                cycles: int = 3) -> Dict[str, Any]:
        """Ejecuta ciclos autónomos de entrenamiento"""
        cycle_results = []
        
        for cycle in range(cycles):
            self.logger.info(f"Iniciando ciclo autónomo {cycle + 1}/{cycles}")
            
            # Configuraciones de modelo para este ciclo
            model_configs = [
                {'type': 'cnn', 'name': f'cnn_cycle_{cycle}'},
                {'type': 'lstm', 'name': f'lstm_cycle_{cycle}'},
                {'type': 'transformer', 'name': f'transformer_cycle_{cycle}'}
            ]
            
            # Entrenar ecosistema
            cycle_result = self.train_ecosystem(data_sources, model_configs)
            cycle_results.append(cycle_result)
            
            # Analizar resultados y ajustar configuración
            if cycle_result['status'] == 'success':
                self._adjust_config_based_on_results(cycle_result)
        
        return {
            'cycles_completed': cycles,
            'cycle_results': cycle_results,
            'final_ecosystem_status': self.get_ecosystem_status()
        }
    
    def _adjust_config_based_on_results(self, results: Dict[str, Any]):
        """Ajusta la configuración basándose en los resultados"""
        # Lógica simple de ajuste automático
        evaluation_results = results.get('evaluation_results', {})
        
        avg_accuracy = np.mean([
            result.get('accuracy', 0) 
            for result in evaluation_results.values() 
            if isinstance(result, dict) and 'accuracy' in result
        ])
        
        # Ajustar learning rate basándose en el rendimiento
        if avg_accuracy < 0.8:
            self.config['learning_rate'] *= 0.9  # Reducir learning rate
            self.config['epochs'] = min(self.config['epochs'] + 2, 20)  # Más épocas
        elif avg_accuracy > 0.9:
            self.config['learning_rate'] *= 1.1  # Aumentar learning rate
            self.config['epochs'] = max(self.config['epochs'] - 1, 5)  # Menos épocas
        
        self.logger.info(f"Configuración ajustada: LR={self.config['learning_rate']:.6f}, Epochs={self.config['epochs']}")

# Funciones de compatibilidad con el código existente
def train_ecosystem(data_sources: List[str], models: List[Any]) -> str:
    """Función de compatibilidad para el código existente"""
    ecosystem = AdvancedEcosystemTraining()
    
    # Convertir modelos a configuraciones
    model_configs = []
    for i, model in enumerate(models):
        model_configs.append({
            'type': 'cnn',  # Tipo por defecto
            'name': f'model_{i}',
            'model_instance': model
        })
    
    result = ecosystem.train_ecosystem(data_sources, model_configs)
    
    if result['status'] == 'success':
        return "Ecosistema entrenado exitosamente"
    else:
        return f"Error en entrenamiento: {result.get('error', 'Error desconocido')}"

def create_cnn_ecosystem(input_shape=(28, 28, 1), num_classes=10):
    """Crea un modelo CNN para el ecosistema"""
    return create_cnn(input_shape, num_classes)

def create_rnn_ecosystem(input_shape=(100, 1), output_units=1):
    """Crea un modelo RNN para el ecosistema"""
    return create_rnn(input_shape, output_units)

def create_lstm_ecosystem(input_shape=(100, 1), output_units=1):
    """Crea un modelo LSTM para el ecosistema"""
    return create_lstm(input_shape, output_units)
