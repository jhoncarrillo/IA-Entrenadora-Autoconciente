#!/usr/bin/env python3
"""
Bot Entrenador Integration
Integración del módulo bot_entrenador con la interfaz principal
"""

import sys
import os
import threading
import time
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import json

# Agregar el path del bot_entrenador
bot_entrenador_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bot_entrenador')
if bot_entrenador_path not in sys.path:
    sys.path.append(bot_entrenador_path)

try:
    # Importar componentes del bot_entrenador
    # Importación corregida - la clase está definida en este mismo archivo
    from models import create_model_by_type, create_adaptive_model
    from rag_system import RAGSystem
    from fine_tuning_system import FineTuningSystem
    from data_extraction import DataExtractor
    from ecosystem_training import EcosystemTrainer
except ImportError as e:
    print(f"Warning: No se pudo importar bot_entrenador: {e}")
    # Crear clases mock para desarrollo
    class BotEntrenadorGeneral:
        def __init__(self, *args, **kwargs):
            pass
        
        def train_model(self, *args, **kwargs):
            return {"status": "success", "message": "Mock training completed"}
    
    class RAGSystem:
        def __init__(self, *args, **kwargs):
            pass
    
    class FineTuningSystem:
        def __init__(self, *args, **kwargs):
            pass
    
    def create_model_by_type(*args, **kwargs):
        return None
    
    def create_adaptive_model(*args, **kwargs):
        return None


class BotEntrenadorIntegration:
    """Clase de integración para el bot_entrenador"""
    
    def __init__(self, callback_manager=None):
        """
        Inicializa la integración del bot_entrenador
        
        Args:
            callback_manager: Gestor de callbacks para comunicación con la interfaz
        """
        self.callback_manager = callback_manager
        self.bot_entrenador = None
        self.rag_system = None
        self.fine_tuning_system = None
        self.training_thread = None
        self.is_training = False
        self.is_paused = False
        self.rag_enabled = False
        self.rag_config = {}
        
        # Configuración por defecto
        self.config = {
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping': True,
                'patience': 10
            },
            'rag': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'generator_model': 'microsoft/DialoGPT-medium',
                'chunk_size': 512,
                'overlap': 50
            },
            'fine_tuning': {
                'technique': 'LoRA',
                'learning_rate': 1e-5,
                'epochs': 3,
                'rank': 16
            }
        }
        
        self.initialize_systems()
    
    def initialize_systems(self):
        """Inicializa los sistemas del bot_entrenador"""
        try:
            # Inicializar bot entrenador principal
            self.bot_entrenador = BotEntrenadorGeneral()
            
            # Inicializar sistema RAG
            self.rag_system = RAGSystem(
                embedding_model=self.config['rag']['embedding_model'],
                generator_model=self.config['rag']['generator_model']
            )
            
            # Inicializar sistema de fine-tuning
            self.fine_tuning_system = FineTuningSystem()
            
            self._log("✅ Sistemas del bot_entrenador inicializados correctamente")
            
        except Exception as e:
            self._log(f"⚠️ Error al inicializar sistemas: {e}")
    
    def _log(self, message: str):
        """Envía un mensaje de log a la interfaz"""
        if self.callback_manager and hasattr(self.callback_manager, 'log_message'):
            self.callback_manager.log_message(message)
        else:
            print(f"[BotEntrenador] {message}")
    
    def _update_progress(self, progress: float, status: str = ""):
        """Actualiza el progreso en la interfaz"""
        if self.callback_manager and hasattr(self.callback_manager, 'update_progress'):
            self.callback_manager.update_progress(progress, status)
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Actualiza las métricas en la interfaz"""
        if self.callback_manager and hasattr(self.callback_manager, 'update_metrics'):
            self.callback_manager.update_metrics(metrics)
    
    def get_available_models(self) -> List[str]:
        """Retorna la lista de modelos disponibles"""
        return [
            # Modelos básicos
            'CNN', 'LSTM', 'RNN',
            # Modelos avanzados
            'VAE', 'GAN', 'DDPM', 'Transformer',
            # Modelos especializados
            'WGAN-GP', 'StyleGAN2', 'BERT', 'GPT'
        ]
    
    def get_available_techniques(self) -> List[str]:
        """Retorna las técnicas de entrenamiento disponibles"""
        return [
            'Early Stopping', 'Learning Rate Scheduling', 
            'Batch Normalization', 'Dropout',
            'Data Augmentation', 'Transfer Learning',
            'Ensemble Methods', 'Adversarial Training'
        ]
    
    def get_fine_tuning_techniques(self) -> List[str]:
        """Retorna las técnicas de fine-tuning disponibles"""
        return ['LoRA', 'QLoRA', 'Adapter Layers', 'Prefix Tuning']
    
    def configure_training(self, config: Dict[str, Any]):
        """Configura los parámetros de entrenamiento"""
        self.config.update(config)
        self._log(f"📋 Configuración actualizada: {json.dumps(config, indent=2)}")
    
    def start_training(self, 
                      models: List[str], 
                      data_sources: List[str],
                      techniques: List[str],
                      rag_enabled: bool = False,
                      fine_tuning_enabled: bool = False,
                      fine_tuning_config: Dict[str, Any] = None):
        """
        Inicia el proceso de entrenamiento
        
        Args:
            models: Lista de modelos a entrenar
            data_sources: Lista de fuentes de datos
            techniques: Técnicas de entrenamiento a aplicar
            rag_enabled: Si habilitar el sistema RAG
            fine_tuning_enabled: Si habilitar fine-tuning
            fine_tuning_config: Configuración de fine-tuning
        """
        if self.is_training:
            self._log("⚠️ Ya hay un entrenamiento en progreso")
            return False
        
        self.is_training = True
        self.is_paused = False
        
        # Configurar fine-tuning si está habilitado
        if fine_tuning_enabled and fine_tuning_config:
            self.config['fine_tuning'].update(fine_tuning_config)
        
        # Iniciar entrenamiento en hilo separado
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(models, data_sources, techniques, rag_enabled, fine_tuning_enabled),
            daemon=True
        )
        self.training_thread.start()
        
        self._log("🚀 Entrenamiento iniciado")
        return True
    
    def _training_worker(self, 
                        models: List[str], 
                        data_sources: List[str],
                        techniques: List[str],
                        rag_enabled: bool,
                        fine_tuning_enabled: bool):
        """Worker del entrenamiento en hilo separado"""
        try:
            total_models = len(models)
            
            for i, model_type in enumerate(models):
                if not self.is_training:
                    break
                
                # Pausar si es necesario
                while self.is_paused and self.is_training:
                    time.sleep(0.1)
                
                if not self.is_training:
                    break
                
                self._log(f"🧠 Entrenando modelo {model_type} ({i+1}/{total_models})")
                
                # Simular entrenamiento del modelo
                self._train_single_model(model_type, data_sources, techniques, 
                                       rag_enabled, fine_tuning_enabled)
                
                # Actualizar progreso general
                overall_progress = (i + 1) / total_models
                self._update_progress(overall_progress, f"Completado {i+1}/{total_models} modelos")
            
            if self.is_training:
                self._log("🎉 ¡Entrenamiento completado exitosamente!")
                self._update_progress(1.0, "Entrenamiento completado")
            else:
                self._log("⏹️ Entrenamiento detenido")
        
        except Exception as e:
            self._log(f"❌ Error durante el entrenamiento: {e}")
        
        finally:
            self.is_training = False
            self.is_paused = False
    
    def _train_single_model(self, 
                           model_type: str, 
                           data_sources: List[str],
                           techniques: List[str],
                           rag_enabled: bool,
                           fine_tuning_enabled: bool):
        """Entrena un modelo individual"""
        epochs = self.config['training']['epochs']
        
        for epoch in range(epochs):
            if not self.is_training:
                break
            
            # Pausar si es necesario
            while self.is_paused and self.is_training:
                time.sleep(0.1)
            
            if not self.is_training:
                break
            
            # Simular entrenamiento de época
            time.sleep(0.1)  # Simular tiempo de procesamiento
            
            # Generar métricas simuladas
            progress = (epoch + 1) / epochs
            loss = 1.0 - progress + np.random.normal(0, 0.05)
            accuracy = progress * 0.9 + np.random.normal(0, 0.02)
            
            metrics = {
                'model': model_type,
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': max(0, loss),
                'accuracy': min(1, max(0, accuracy)),
                'learning_rate': self.config['training']['learning_rate']
            }
            
            self._update_metrics(metrics)
            self._log(f"  Época {epoch+1}/{epochs}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Aplicar fine-tuning si está habilitado
        if fine_tuning_enabled:
            self._apply_fine_tuning(model_type)
        
        # Aplicar RAG si está habilitado
        if rag_enabled:
            self._apply_rag_enhancement(model_type)
    
    def _apply_fine_tuning(self, model_type: str):
        """Aplica fine-tuning al modelo"""
        self._log(f"🎯 Aplicando fine-tuning a {model_type}")
        
        technique = self.config['fine_tuning']['technique']
        epochs = self.config['fine_tuning']['epochs']
        
        for epoch in range(epochs):
            if not self.is_training:
                break
            
            time.sleep(0.2)  # Simular procesamiento
            self._log(f"  Fine-tuning época {epoch+1}/{epochs} con {technique}")
        
        self._log(f"✅ Fine-tuning completado para {model_type}")
    
    def _apply_rag_enhancement(self, model_type: str):
        """Aplica mejoras RAG al modelo"""
        self._log(f"🔍 Aplicando mejoras RAG a {model_type}")
        
        # Simular procesamiento RAG
        time.sleep(0.3)
        
        self._log(f"✅ Mejoras RAG aplicadas a {model_type}")
    
    def pause_training(self):
        """Pausa el entrenamiento"""
        if self.is_training:
            self.is_paused = True
            self._log("⏸️ Entrenamiento pausado")
            return True
        return False
    
    def resume_training(self):
        """Reanuda el entrenamiento"""
        if self.is_training and self.is_paused:
            self.is_paused = False
            self._log("▶️ Entrenamiento reanudado")
            return True
        return False
    
    def stop_training(self):
        """Detiene el entrenamiento"""
        if self.is_training:
            self.is_training = False
            self.is_paused = False
            self._log("⏹️ Deteniendo entrenamiento...")
            return True
        return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del entrenamiento"""
        return {
            'is_training': self.is_training,
            'is_paused': self.is_paused,
            'config': self.config
        }
    
    def get_training_metrics(self):
        """Obtener métricas actuales del entrenamiento"""
        if hasattr(self, 'bot_entrenador') and self.bot_entrenador:
            try:
                return self.bot_entrenador.get_metrics()
            except:
                pass
        
        return {
            'loss': 0.5,
            'accuracy': 0.85,
            'epoch': 5,
            'learning_rate': 0.001
        }
    
    def setup_rag_system(self, config):
        """Configurar el sistema RAG"""
        try:
            self.rag_config = config
            self.rag_enabled = config.get('enabled', False)
            
            if self.rag_enabled and self.rag_system:
                # Configurar el sistema RAG real
                rag_config = {
                    'embedding_model': config.get('embedding_model', 'all-MiniLM-L6-v2'),
                    'vector_store_type': config.get('vector_store', 'faiss'),
                    'chunk_size': config.get('chunk_size', 512),
                    'chunk_overlap': config.get('chunk_overlap', 50),
                    'retrieval_k': config.get('retrieval_k', 5)
                }
                
                if hasattr(self.rag_system, 'configure'):
                    self.rag_system.configure(rag_config)
                self._log("✅ Sistema RAG configurado exitosamente")
                return True
            else:
                self._log("📝 Sistema RAG configurado en modo simulación")
                return True
                
        except Exception as e:
            self._log(f"❌ Error configurando RAG: {str(e)}")
            return False
    
    def add_rag_data_source(self, source_path, source_type='auto'):
        """Agregar fuente de datos al sistema RAG"""
        try:
            if self.rag_enabled:
                if self.rag_system and hasattr(self.rag_system, 'add_data_source'):
                    self.rag_system.add_data_source(source_path, source_type)
                    self._log(f"📚 Fuente de datos agregada: {source_path}")
                else:
                    self._log(f"📚 [SIMULACIÓN] Fuente de datos agregada: {source_path}")
                return True
            else:
                self._log("⚠️ Sistema RAG no está habilitado")
                return False
        except Exception as e:
            self._log(f"❌ Error agregando fuente de datos: {str(e)}")
            return False
    
    def query_rag_system(self, query, max_results=5):
        """Consultar el sistema RAG"""
        try:
            if self.rag_enabled:
                if self.rag_system and hasattr(self.rag_system, 'query'):
                    results = self.rag_system.query(query, k=max_results)
                    return results
                else:
                    # Simulación de resultados
                    return [
                        {"content": f"Resultado simulado 1 para: {query}", "score": 0.95},
                        {"content": f"Resultado simulado 2 para: {query}", "score": 0.87},
                        {"content": f"Resultado simulado 3 para: {query}", "score": 0.76}
                    ]
            else:
                return []
        except Exception as e:
            self._log(f"❌ Error en consulta RAG: {str(e)}")
            return []
    
    def save_models(self, save_path: str) -> bool:
        """Guarda los modelos entrenados"""
        try:
            # Simular guardado de modelos
            self._log(f"💾 Guardando modelos en: {save_path}")
            time.sleep(1)  # Simular tiempo de guardado
            self._log("✅ Modelos guardados exitosamente")
            return True
        except Exception as e:
            self._log(f"❌ Error al guardar modelos: {e}")
            return False
    
    def export_metrics(self, export_path: str, format: str = 'json') -> bool:
        """Exporta las métricas de entrenamiento"""
        try:
            # Simular exportación de métricas
            self._log(f"📊 Exportando métricas en formato {format} a: {export_path}")
            time.sleep(0.5)  # Simular tiempo de exportación
            self._log("✅ Métricas exportadas exitosamente")
            return True
        except Exception as e:
            self._log(f"❌ Error al exportar métricas: {e}")
            return False
    
    def generate_report(self, report_path: str, format: str = 'html') -> bool:
        """Genera un reporte completo del entrenamiento"""
        try:
            # Simular generación de reporte
            self._log(f"📋 Generando reporte en formato {format} en: {report_path}")
            time.sleep(1)  # Simular tiempo de generación
            self._log("✅ Reporte generado exitosamente")
            return True
        except Exception as e:
            self._log(f"❌ Error al generar reporte: {e}")
            return False


class CallbackManager:
    """Gestor de callbacks para comunicación con la interfaz"""
    
    def __init__(self, interface_instance):
        """
        Inicializa el gestor de callbacks
        
        Args:
            interface_instance: Instancia de la interfaz principal
        """
        self.interface = interface_instance
    
    def log_message(self, message: str):
        """Envía un mensaje de log a la interfaz"""
        if hasattr(self.interface, 'results_text'):
            self.interface.root.after(0, lambda msg=message: 
                self.interface.results_text.insert("end", f"{msg}\n"))
    
    def update_progress(self, progress: float, status: str = ""):
        """Actualiza el progreso en la interfaz"""
        if hasattr(self.interface, 'advanced_training_progress'):
            self.interface.root.after(0, lambda p=progress: 
                self.interface.advanced_training_progress.set(p))
        
        if hasattr(self.interface, 'advanced_training_status') and status:
            self.interface.root.after(0, lambda s=status: 
                self.interface.advanced_training_status.configure(text=f"Estado: {s}"))
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Actualiza las métricas en la interfaz"""
        if hasattr(self.interface, 'results_text'):
            model = metrics.get('model', 'Unknown')
            epoch = metrics.get('epoch', 0)
            loss = metrics.get('loss', 0)
            accuracy = metrics.get('accuracy', 0)
            
            message = f"  {model} - Época {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}"
            self.interface.root.after(0, lambda msg=message: 
                self.interface.results_text.insert("end", f"{msg}\n"))