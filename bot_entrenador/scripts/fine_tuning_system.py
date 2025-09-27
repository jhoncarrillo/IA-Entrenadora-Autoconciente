"""
Sistema de Fine-Tuning Automático para Modelos Preentrenados
Implementa fine-tuning inteligente con optimización autónoma y monitoreo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import json
import os
from pathlib import Path
import pickle
from datetime import datetime
import time
import copy
from collections import defaultdict
import threading
import queue

# Transformers y optimización
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model, LoraConfig, TaskType, PeftModel,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig

# Métricas y evaluación
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb

# Optimización de hiperparámetros
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna no disponible, optimización de hiperparámetros limitada")

logger = logging.getLogger(__name__)

class AdaptiveDataset(Dataset):
    """Dataset adaptativo que se ajusta automáticamente al tipo de tarea"""
    
    def __init__(self, data: List[Dict], tokenizer, task_type: str = 'classification',
                 max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length
        
        # Detectar automáticamente el formato de datos
        self._analyze_data_format()
        
    def _analyze_data_format(self):
        """Analizar formato de datos automáticamente"""
        if not self.data:
            raise ValueError("Dataset vacío")
        
        sample = self.data[0]
        
        # Detectar campos comunes
        self.text_field = None
        self.label_field = None
        self.target_field = None
        
        # Buscar campos de texto
        text_candidates = ['text', 'input', 'sentence', 'content', 'prompt', 'question']
        for field in text_candidates:
            if field in sample:
                self.text_field = field
                break
        
        # Buscar campos de etiquetas
        label_candidates = ['label', 'target', 'class', 'category', 'output', 'answer']
        for field in label_candidates:
            if field in sample:
                self.label_field = field
                break
        
        # Para tareas de generación
        if self.task_type in ['generation', 'causal_lm']:
            target_candidates = ['target', 'output', 'response', 'answer', 'completion']
            for field in target_candidates:
                if field in sample:
                    self.target_field = field
                    break
        
        logger.info(f"Campos detectados - Texto: {self.text_field}, Etiqueta: {self.label_field}, Target: {self.target_field}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task_type == 'classification':
            return self._prepare_classification_item(item)
        elif self.task_type == 'generation':
            return self._prepare_generation_item(item)
        elif self.task_type == 'causal_lm':
            return self._prepare_causal_lm_item(item)
        else:
            return self._prepare_generic_item(item)
    
    def _prepare_classification_item(self, item):
        """Preparar item para clasificación"""
        text = item.get(self.text_field, str(item))
        label = item.get(self.label_field, 0)
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _prepare_generation_item(self, item):
        """Preparar item para generación"""
        input_text = item.get(self.text_field, str(item))
        target_text = item.get(self.target_field, "")
        
        # Combinar input y target para entrenamiento
        full_text = f"{input_text} {self.tokenizer.sep_token} {target_text}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }
    
    def _prepare_causal_lm_item(self, item):
        """Preparar item para modelado de lenguaje causal"""
        text = item.get(self.text_field, str(item))
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }
    
    def _prepare_generic_item(self, item):
        """Preparar item genérico"""
        text = str(item)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class AdvancedTrainer:
    """Trainer avanzado con capacidades de optimización automática"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.training_history = defaultdict(list)
        self.best_metrics = {}
        self.optimization_history = []
        
        # Configuración de entrenamiento
        self.task_type = config.get('task_type', 'classification')
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.use_lora = config.get('use_lora', True)
        self.use_quantization = config.get('use_quantization', False)
        self.auto_optimize = config.get('auto_optimize', True)
        
        # Métricas y callbacks
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.metric_for_best_model = config.get('metric_for_best_model', 'eval_loss')
        
        # Inicializar modelo y tokenizer
        self._initialize_model()
        
    def _initialize_model(self):
        """Inicializar modelo y tokenizer"""
        try:
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configurar pad token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configuración de cuantización
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            
            # Cargar modelo según el tipo de tarea
            if self.task_type == 'classification':
                num_labels = self.config.get('num_labels', 2)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels,
                    quantization_config=quantization_config
                )
            elif self.task_type in ['generation', 'causal_lm']:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config
                )
            
            # Configurar LoRA si está habilitado
            if self.use_lora:
                self._setup_lora()
            
            logger.info(f"Modelo {self.model_name} inicializado para tarea {self.task_type}")
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise
    
    def _setup_lora(self):
        """Configurar LoRA para fine-tuning eficiente"""
        try:
            # Preparar modelo para entrenamiento con cuantización
            if self.use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Configuración de LoRA
            if self.task_type == 'classification':
                task_type = TaskType.SEQ_CLS
                target_modules = ["query", "value"]
            elif self.task_type in ['generation', 'causal_lm']:
                task_type = TaskType.CAUSAL_LM
                target_modules = ["q_proj", "v_proj"]
            else:
                task_type = TaskType.FEATURE_EXTRACTION
                target_modules = ["query", "value"]
            
            lora_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=self.config.get('lora_r', 8),
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                target_modules=target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("LoRA configurado exitosamente")
            
        except Exception as e:
            logger.error(f"Error configurando LoRA: {e}")
            self.use_lora = False
    
    def prepare_data(self, train_data: List[Dict], val_data: List[Dict] = None,
                    test_data: List[Dict] = None, val_split: float = 0.2):
        """Preparar datos para entrenamiento"""
        
        # Crear dataset de entrenamiento
        train_dataset = AdaptiveDataset(
            train_data, self.tokenizer, self.task_type,
            max_length=self.config.get('max_length', 512)
        )
        
        # Crear dataset de validación
        if val_data is not None:
            val_dataset = AdaptiveDataset(
                val_data, self.tokenizer, self.task_type,
                max_length=self.config.get('max_length', 512)
            )
        elif val_split > 0:
            # Dividir datos de entrenamiento
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            val_dataset = None
        
        # Crear dataset de prueba
        test_dataset = None
        if test_data is not None:
            test_dataset = AdaptiveDataset(
                test_data, self.tokenizer, self.task_type,
                max_length=self.config.get('max_length', 512)
            )
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset=None, **training_args):
        """Entrenar modelo con configuración optimizada"""
        
        # Configuración de entrenamiento por defecto
        default_args = {
            'output_dir': './fine_tuned_model',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_dir': './logs',
            'logging_steps': 100,
            'evaluation_strategy': 'steps',
            'eval_steps': 500,
            'save_strategy': 'steps',
            'save_steps': 500,
            'load_best_model_at_end': True,
            'metric_for_best_model': self.metric_for_best_model,
            'greater_is_better': False if 'loss' in self.metric_for_best_model else True,
            'save_total_limit': 3,
            'seed': 42,
            'fp16': torch.cuda.is_available(),
            'dataloader_pin_memory': False,
            'remove_unused_columns': False
        }
        
        # Actualizar con argumentos proporcionados
        default_args.update(training_args)
        
        # Crear argumentos de entrenamiento
        training_arguments = TrainingArguments(**default_args)
        
        # Función de métricas
        def compute_metrics(eval_pred):
            return self._compute_metrics(eval_pred)
        
        # Callbacks
        callbacks = []
        if val_dataset is not None and self.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience
            ))
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
            callbacks=callbacks
        )
        
        # Entrenar
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # Guardar métricas
        self.training_history['training_time'].append(training_time)
        self.training_history['train_loss'].append(train_result.training_loss)
        
        # Evaluación final
        if val_dataset is not None:
            eval_result = trainer.evaluate()
            self.training_history['eval_metrics'].append(eval_result)
            
            # Actualizar mejores métricas
            for metric, value in eval_result.items():
                if metric not in self.best_metrics or self._is_better_metric(metric, value):
                    self.best_metrics[metric] = value
        
        # Guardar modelo
        trainer.save_model()
        self.tokenizer.save_pretrained(default_args['output_dir'])
        
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return trainer, train_result
    
    def _compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        predictions, labels = eval_pred
        
        if self.task_type == 'classification':
            # Para clasificación
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        elif self.task_type in ['generation', 'causal_lm']:
            # Para generación (perplexity)
            shift_logits = predictions[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            perplexity = torch.exp(loss)
            
            return {
                'perplexity': perplexity.item()
            }
        
        else:
            # Métricas genéricas
            mse = np.mean((predictions - labels) ** 2)
            return {'mse': mse}
    
    def _is_better_metric(self, metric_name: str, new_value: float) -> bool:
        """Determinar si una nueva métrica es mejor"""
        if metric_name not in self.best_metrics:
            return True
        
        current_best = self.best_metrics[metric_name]
        
        # Métricas donde menor es mejor
        if any(term in metric_name.lower() for term in ['loss', 'error', 'perplexity']):
            return new_value < current_best
        else:
            # Métricas donde mayor es mejor
            return new_value > current_best
    
    def optimize_hyperparameters(self, train_dataset, val_dataset, n_trials: int = 20):
        """Optimizar hiperparámetros usando Optuna"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna no disponible, usando configuración por defecto")
            return self.config
        
        def objective(trial):
            # Sugerir hiperparámetros
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            num_epochs = trial.suggest_int('num_epochs', 2, 5)
            warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
            weight_decay = trial.suggest_float('weight_decay', 0.0, 0.3)
            
            if self.use_lora:
                lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
                lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
                lora_dropout = trial.suggest_float('lora_dropout', 0.05, 0.3)
                
                # Reconfigurar LoRA
                self.config.update({
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout
                })
                self._initialize_model()
            
            # Configurar entrenamiento
            training_args = {
                'learning_rate': learning_rate,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'num_train_epochs': num_epochs,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'output_dir': f'./optuna_trial_{trial.number}',
                'logging_steps': 50,
                'eval_steps': 100,
                'save_steps': 100,
                'evaluation_strategy': 'steps'
            }
            
            try:
                # Entrenar modelo
                trainer, _ = self.train(train_dataset, val_dataset, **training_args)
                
                # Evaluar
                eval_result = trainer.evaluate()
                
                # Retornar métrica objetivo
                if self.task_type == 'classification':
                    return eval_result.get('eval_f1', eval_result.get('eval_accuracy', 0))
                else:
                    return -eval_result.get('eval_loss', float('inf'))
                
            except Exception as e:
                logger.error(f"Error en trial {trial.number}: {e}")
                return float('-inf')
        
        # Crear estudio
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Obtener mejores parámetros
        best_params = study.best_params
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        })
        
        logger.info(f"Optimización completada. Mejores parámetros: {best_params}")
        
        return best_params
    
    def evaluate_model(self, test_dataset):
        """Evaluar modelo en conjunto de prueba"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Crear trainer para evaluación
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        
        # Evaluar
        eval_result = trainer.evaluate(test_dataset)
        
        logger.info(f"Resultados de evaluación: {eval_result}")
        
        return eval_result
    
    def save_model(self, path: str):
        """Guardar modelo entrenado"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo y tokenizer
        if self.use_lora:
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        
        self.tokenizer.save_pretrained(path)
        
        # Guardar configuración y métricas
        metadata = {
            'config': self.config,
            'training_history': dict(self.training_history),
            'best_metrics': self.best_metrics,
            'optimization_history': self.optimization_history,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'use_lora': self.use_lora,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo guardado en {path}")
    
    def load_model(self, path: str):
        """Cargar modelo entrenado"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {path}")
        
        # Cargar metadata
        with open(path / 'training_metadata.json', 'r') as f:
            metadata = json.load(f)
            
            self.config = metadata['config']
            self.training_history = defaultdict(list, metadata['training_history'])
            self.best_metrics = metadata['best_metrics']
            self.optimization_history = metadata['optimization_history']
            self.model_name = metadata['model_name']
            self.task_type = metadata['task_type']
            self.use_lora = metadata['use_lora']
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Cargar modelo
        if self.use_lora:
            # Cargar modelo base y adaptador LoRA
            base_model = AutoModel.from_pretrained(self.model_name)
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            if self.task_type == 'classification':
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
            elif self.task_type in ['generation', 'causal_lm']:
                self.model = AutoModelForCausalLM.from_pretrained(path)
            else:
                self.model = AutoModel.from_pretrained(path)
        
        logger.info(f"Modelo cargado desde {path}")

class AutonomousFineTuningSystem:
    """Sistema de Fine-Tuning autónomo con gestión inteligente"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_trainers = {}
        self.training_queue = queue.Queue()
        self.results_history = []
        
        # Configuración del sistema
        self.max_concurrent_trainings = config.get('max_concurrent_trainings', 2)
        self.auto_optimize = config.get('auto_optimize', True)
        self.save_all_models = config.get('save_all_models', False)
        self.models_dir = Path(config.get('models_dir', './fine_tuned_models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Hilo de procesamiento
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        if config.get('auto_start', True):
            self.start_processing()
    
    def add_training_job(self, job_config: Dict[str, Any]) -> str:
        """Agregar trabajo de fine-tuning a la cola"""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.results_history)}"
        
        job = {
            'job_id': job_id,
            'config': job_config,
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'priority': job_config.get('priority', 1)
        }
        
        self.training_queue.put(job)
        logger.info(f"Trabajo de fine-tuning agregado: {job_id}")
        
        return job_id
    
    def start_processing(self):
        """Iniciar procesamiento de trabajos"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_jobs)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Procesamiento de fine-tuning iniciado")
    
    def stop_processing(self):
        """Detener procesamiento de trabajos"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Procesamiento de fine-tuning detenido")
    
    def _process_jobs(self):
        """Procesar trabajos de fine-tuning"""
        while not self.stop_processing.is_set():
            try:
                # Verificar si hay espacio para nuevos entrenamientos
                if len(self.active_trainers) >= self.max_concurrent_trainings:
                    time.sleep(5)
                    continue
                
                # Obtener siguiente trabajo
                try:
                    job = self.training_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                # Procesar trabajo
                self._execute_training_job(job)
                
            except Exception as e:
                logger.error(f"Error procesando trabajos: {e}")
                time.sleep(5)
    
    def _execute_training_job(self, job: Dict[str, Any]):
        """Ejecutar trabajo de fine-tuning"""
        job_id = job['job_id']
        config = job['config']
        
        try:
            logger.info(f"Iniciando fine-tuning: {job_id}")
            
            # Crear trainer
            trainer = AdvancedTrainer(config)
            self.active_trainers[job_id] = trainer
            
            # Preparar datos
            train_data = config['train_data']
            val_data = config.get('val_data')
            test_data = config.get('test_data')
            
            train_dataset, val_dataset, test_dataset = trainer.prepare_data(
                train_data, val_data, test_data
            )
            
            # Optimizar hiperparámetros si está habilitado
            if self.auto_optimize and config.get('optimize_hyperparams', True):
                best_params = trainer.optimize_hyperparameters(
                    train_dataset, val_dataset,
                    n_trials=config.get('optimization_trials', 10)
                )
                config.update(best_params)
            
            # Entrenar modelo
            start_time = time.time()
            trainer_obj, train_result = trainer.train(train_dataset, val_dataset)
            training_time = time.time() - start_time
            
            # Evaluar en conjunto de prueba si está disponible
            test_results = None
            if test_dataset is not None:
                test_results = trainer.evaluate_model(test_dataset)
            
            # Guardar modelo si está configurado
            model_path = None
            if self.save_all_models or config.get('save_model', True):
                model_path = self.models_dir / job_id
                trainer.save_model(model_path)
            
            # Registrar resultados
            result = {
                'job_id': job_id,
                'status': 'completed',
                'config': config,
                'training_time': training_time,
                'train_result': train_result.metrics if hasattr(train_result, 'metrics') else {},
                'test_results': test_results,
                'best_metrics': trainer.best_metrics,
                'model_path': str(model_path) if model_path else None,
                'completed_at': datetime.now().isoformat()
            }
            
            self.results_history.append(result)
            
            logger.info(f"Fine-tuning completado: {job_id}")
            
        except Exception as e:
            logger.error(f"Error en fine-tuning {job_id}: {e}")
            
            # Registrar error
            error_result = {
                'job_id': job_id,
                'status': 'failed',
                'config': config,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            
            self.results_history.append(error_result)
        
        finally:
            # Limpiar trainer activo
            if job_id in self.active_trainers:
                del self.active_trainers[job_id]
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtener estado de un trabajo"""
        # Buscar en trabajos activos
        if job_id in self.active_trainers:
            return {
                'job_id': job_id,
                'status': 'running',
                'trainer': self.active_trainers[job_id]
            }
        
        # Buscar en historial
        for result in self.results_history:
            if result['job_id'] == job_id:
                return result
        
        return {'job_id': job_id, 'status': 'not_found'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema"""
        completed_jobs = sum(1 for r in self.results_history if r['status'] == 'completed')
        failed_jobs = sum(1 for r in self.results_history if r['status'] == 'failed')
        
        return {
            'active_trainings': len(self.active_trainers),
            'queued_jobs': self.training_queue.qsize(),
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'total_jobs': len(self.results_history),
            'max_concurrent': self.max_concurrent_trainings,
            'auto_optimize': self.auto_optimize
        }
    
    def get_best_models(self, metric: str = 'eval_f1', top_k: int = 5) -> List[Dict]:
        """Obtener mejores modelos según métrica"""
        completed_results = [r for r in self.results_history if r['status'] == 'completed']
        
        # Ordenar por métrica
        def get_metric_value(result):
            if 'best_metrics' in result and metric in result['best_metrics']:
                return result['best_metrics'][metric]
            return float('-inf')
        
        sorted_results = sorted(completed_results, key=get_metric_value, reverse=True)
        
        return sorted_results[:top_k]
    
    def save_system_state(self, path: str):
        """Guardar estado del sistema"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        system_state = {
            'config': self.config,
            'results_history': self.results_history,
            'system_status': self.get_system_status(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path / 'fine_tuning_system_state.json', 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logger.info(f"Estado del sistema guardado en {path}")

def create_fine_tuning_system(config: Dict[str, Any] = None) -> AutonomousFineTuningSystem:
    """Factory function para crear sistema de fine-tuning"""
    
    default_config = {
        'max_concurrent_trainings': 2,
        'auto_optimize': True,
        'save_all_models': False,
        'models_dir': './fine_tuned_models',
        'auto_start': True
    }
    
    if config:
        default_config.update(config)
    
    return AutonomousFineTuningSystem(default_config)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🎯 Creando Sistema de Fine-Tuning Autónomo...")
    
    # Crear sistema
    ft_system = create_fine_tuning_system()
    
    # Configuración de trabajo de ejemplo
    job_config = {
        'model_name': 'bert-base-uncased',
        'task_type': 'classification',
        'num_labels': 2,
        'train_data': [
            {'text': 'Este es un ejemplo positivo', 'label': 1},
            {'text': 'Este es un ejemplo negativo', 'label': 0}
        ],
        'use_lora': True,
        'optimize_hyperparams': True,
        'save_model': True
    }
    
    # Agregar trabajo
    job_id = ft_system.add_training_job(job_config)
    print(f"Trabajo agregado: {job_id}")
    
    # Obtener estado del sistema
    status = ft_system.get_system_status()
    print(f"Estado del sistema: {status}")
    
    print("Sistema de Fine-Tuning Autónomo creado exitosamente! 🚀")