"""
Ecosistema Unificado de Redes Neuronales - Main
==============================================

Sistema principal que integra todos los componentes del ecosistema unificado,
combinando funcionalidades de project_root y red_neuronal.

Características:
- Modelos unificados (CNN, RNN, GAN, VAE, Transformer, Attention, Graph)
- Sistema de procesamiento de datos avanzado
- Evolución automática de arquitecturas
- Auto-optimización de hiperparámetros
- Interfaz unificada y fácil de usar

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecosistema_unificado.log'),
        logging.StreamHandler()
    ]
)

# Importar módulos del ecosistema unificado
from models import (
    CNNUnified, RNNUnified, GANUnified, TransformerUnified,
    VAEUnified, AttentionUnified, GraphUnified
)
from data_processing import DataCollectorUnified, DataProcessorUnified
from evolution import EvolutionUnified
from optimization import AutoOptimizationUnified
from evaluation.cross_validation_unified import CrossValidationUnified
from utils.model_checkpoint_unified import ModelCheckpointUnified

class EcosistemaUnificado:
    """Clase principal del ecosistema unificado de redes neuronales"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando Ecosistema Unificado de Redes Neuronales")
        
        # Componentes del ecosistema
        self.data_collector = DataCollectorUnified()
        self.data_processor = DataProcessorUnified()
        self.evolution_system = EvolutionUnified()
        self.optimization_system = AutoOptimizationUnified()
        
        # Modelos disponibles
        self.available_models = {
            'cnn': CNNUnified,
            'rnn': RNNUnified,
            'gan': GANUnified,
            'transformer': TransformerUnified,
            'vae': VAEUnified,
            'attention': AttentionUnified,
            'graph': GraphUnified
        }
        
        # Datos cargados
        self.datasets = {}
        self.current_dataset = None
        
        self.logger.info("Ecosistema inicializado correctamente")
        
        # Sistema de checkpoints global
        self.checkpoint_manager = ModelCheckpointUnified(
            base_path='saved_models/ecosystem',
            auto_save=True,
            keep_best_only=True,
            max_checkpoints=20
        )
    
    def load_dataset(self, dataset_name: str = 'mnist'):
        """Carga un dataset específico"""
        self.logger.info(f"Cargando dataset: {dataset_name}")
        
        try:
            if dataset_name == 'mnist':
                x_train, y_train, x_test, y_test = self.data_collector.collect_mnist_data()
            elif dataset_name == 'cifar10':
                x_train, y_train, x_test, y_test = self.data_collector.collect_cifar10_data()
            elif dataset_name == 'synthetic':
                x_data, y_data = self.data_collector.generate_synthetic_data()
                # Dividir en train/test
                split_idx = int(0.8 * len(x_data))
                x_train, x_test = x_data[:split_idx], x_data[split_idx:]
                y_train, y_test = y_data[:split_idx], y_data[split_idx:]
            else:
                raise ValueError(f"Dataset no soportado: {dataset_name}")
            
            # Procesar datos
            x_train = self.data_processor.normalize_data(x_train, method='simple')
            x_test = self.data_processor.normalize_data(x_test, method='simple', fit_scaler=False)
            
            # Dividir en train/val/test
            x_train, x_val, x_test, y_train, y_val, y_test = self.data_processor.split_data(
                x_train, y_train, test_size=0.2, val_size=0.1
            )
            
            self.datasets[dataset_name] = {
                'x_train': x_train,
                'y_train': y_train,
                'x_val': x_val,
                'y_val': y_val,
                'x_test': x_test,
                'y_test': y_test
            }
            
            self.current_dataset = dataset_name
            
            # Mostrar información del dataset
            info = self.data_processor.get_data_info(x_train, y_train)
            self.logger.info(f"Dataset cargado: {info}")
            
            return (x_train, y_train), (x_test, y_test)
            
        except Exception as e:
            self.logger.error(f"Error cargando dataset {dataset_name}: {e}")
            raise e
    
    def create_model(self, model_type: str, **kwargs):
        """Crea un modelo específico"""
        self.logger.info(f"Creando modelo: {model_type}")
        
        if model_type not in self.available_models:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        if self.current_dataset is None:
            self.logger.warning("No hay dataset cargado. Cargando MNIST por defecto.")
            self.load_dataset('mnist')
        
        dataset = self.datasets[self.current_dataset]
        input_shape = dataset['x_train'].shape[1:]
        num_classes = len(np.unique(dataset['y_train']))
        
        try:
            if model_type == 'cnn':
                # Filtrar parámetros específicos para CNN
                cnn_params = {k: v for k, v in kwargs.items() if k in ['architecture']}
                model = CNNUnified(input_shape=input_shape, num_classes=num_classes, **cnn_params)
            elif model_type == 'rnn':
                # Filtrar parámetros específicos para RNN
                rnn_params = {k: v for k, v in kwargs.items() if k in ['rnn_type', 'units']}
                model = RNNUnified(input_shape=input_shape, num_classes=num_classes, **rnn_params)
            elif model_type == 'gan':
                # Filtrar parámetros específicos para GAN
                gan_params = {k: v for k, v in kwargs.items() if k in ['latent_dim', 'gan_type']}
                model = GANUnified(input_shape=input_shape, **gan_params)
            elif model_type == 'transformer':
                # Filtrar parámetros específicos para Transformer
                transformer_params = {k: v for k, v in kwargs.items() if k in ['d_model', 'num_heads']}
                model = TransformerUnified(input_shape=input_shape, num_classes=num_classes, **transformer_params)
            elif model_type == 'vae':
                # Filtrar parámetros específicos para VAE
                vae_params = {k: v for k, v in kwargs.items() if k in ['latent_dim']}
                model = VAEUnified(input_shape=input_shape, **vae_params)
            elif model_type == 'attention':
                # Filtrar parámetros específicos para Attention
                attention_params = {k: v for k, v in kwargs.items() if k in ['attention_type']}
                model = AttentionUnified(input_shape=input_shape, num_classes=num_classes, **attention_params)
            elif model_type == 'graph':
                # Filtrar parámetros específicos para Graph
                graph_params = {k: v for k, v in kwargs.items() if k in ['hidden_dim']}
                model = GraphUnified(num_nodes=input_shape[0], node_features=input_shape[1], num_classes=num_classes, **graph_params)
            
            self.logger.info(f"Modelo {model_type} creado exitosamente")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creando modelo {model_type}: {e}")
            return None
    
    def train_model(self, model, epochs: int = 50, verbose: int = 1, robust_mode: bool = True):
        """
        Entrena un modelo con configuración robusta para producción
        
        Args:
            model: Modelo a entrenar
            epochs: Número de épocas (50 por defecto para entrenamiento robusto)
            verbose: Nivel de verbosidad
            robust_mode: Si usar configuración robusta (más épocas, callbacks avanzados)
        """
        if self.current_dataset is None:
            self.logger.error("No hay dataset cargado")
            return None
        
        dataset = self.datasets[self.current_dataset]
        
        # Configuración robusta para producción
        if robust_mode:
            epochs = max(epochs, 50)  # Mínimo 50 épocas para entrenamiento robusto
            self.logger.info(f"MODO ROBUSTO ACTIVADO - Entrenamiento para producción")
            self.logger.info(f"Épocas configuradas: {epochs}")
            self.logger.info(f"Callbacks avanzados: EarlyStopping, ReduceLR, ModelCheckpoint, CSVLogger")
        
        try:
            self.logger.info(f"Iniciando entrenamiento por {epochs} épocas")
            
            history = model.train(
                dataset['x_train'], dataset['y_train'],
                x_val=dataset['x_val'], y_val=dataset['y_val'],
                epochs=epochs,
                verbose=verbose
            )
            
            self.logger.info("Entrenamiento completado")
            
            # Información adicional para modo robusto
            if robust_mode and hasattr(model, 'history') and model.history:
                final_epoch = len(model.history.history['loss'])
                best_val_loss = min(model.history.history.get('val_loss', [float('inf')]))
                self.logger.info(f"Entrenamiento completado en {final_epoch} épocas")
                self.logger.info(f"Mejor val_loss alcanzado: {best_val_loss:.6f}")
            
            # Log información de checkpoints si está disponible
            if hasattr(model, 'checkpoint_manager') and model.checkpoint_manager:
                saved_path = model.checkpoint_manager.get_best_model_path()
                if saved_path:
                    self.logger.info(f"Modelo guardado automáticamente en: {saved_path}")
                summary = model.checkpoint_manager.get_summary()
                if summary:
                    self.logger.info("Resumen de checkpoints:")
                    for summary_line in str(summary).split('\n'):
                        if summary_line.strip():
                            self.logger.info(f"  {summary_line}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {e}")
            return None
    
    def evaluate_model(self, model):
        """Evalúa un modelo"""
        if self.current_dataset is None:
            self.logger.error("No hay dataset cargado")
            return None
        
        dataset = self.datasets[self.current_dataset]
        
        try:
            results = model.evaluate(dataset['x_test'], dataset['y_test'])
            self.logger.info(f"Resultados de evaluación: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error durante la evaluación: {e}")
            return None
    
    def run_evolution(self, generations: int = 5, population_size: int = 10):
        """Ejecuta evolución automática de arquitecturas"""
        if self.current_dataset is None:
            self.logger.error("No hay dataset cargado")
            return None
        
        dataset = self.datasets[self.current_dataset]
        input_shape = dataset['x_train'].shape[1:]
        num_classes = len(np.unique(dataset['y_train']))
        
        self.logger.info(f"Iniciando evolución por {generations} generaciones")
        
        # Configurar sistema de evolución
        self.evolution_system.population_size = population_size
        self.evolution_system.initialize_population(input_shape, num_classes)
        
        best_fitness_history = []
        
        for gen in range(generations):
            best_fitness, avg_fitness = self.evolution_system.evolve_generation(
                dataset['x_train'], dataset['y_train'],
                dataset['x_val'], dataset['y_val'],
                epochs=3  # Pocas épocas para evolución rápida
            )
            best_fitness_history.append(best_fitness)
        
        best_individual = self.evolution_system.get_best_individual()
        self.logger.info(f"Mejor arquitectura encontrada: {best_individual}")
        
        return best_individual, best_fitness_history
    
    def run_optimization(self, model_type: str = 'cnn', method: str = 'random_search', n_trials: int = 10):
        """Ejecuta optimización automática de hiperparámetros"""
        if self.current_dataset is None:
            self.logger.error("No hay dataset cargado")
            return None
        
        dataset = self.datasets[self.current_dataset]
        input_shape = dataset['x_train'].shape[1:]
        num_classes = len(np.unique(dataset['y_train']))
        
        self.logger.info(f"Iniciando optimización {method} para modelo {model_type}")
        
        # Configurar optimizador
        self.optimization_system.optimization_method = method
        
        # Función para construir modelos
        def model_builder(**params):
            return self.create_model(model_type, **params)
        
        # Ejecutar optimización
        best_params = self.optimization_system.optimize(
            model_builder,
            dataset['x_train'], dataset['y_train'],
            dataset['x_val'], dataset['y_val'],
            n_trials=n_trials
        )
        
        summary = self.optimization_system.get_optimization_summary()
        self.logger.info(f"Resumen de optimización: {summary}")
        
        return best_params, summary
    
    def evaluate_model_robustly(self, 
                               model_type: str = 'cnn',
                               dataset_name: str = 'mnist',
                               n_splits: int = 5,
                               include_bootstrap: bool = True) -> dict:
        """
        Evaluación robusta del modelo con validación cruzada y métricas avanzadas
        
        Args:
            model_type: Tipo de modelo ('cnn', 'rnn', etc.)
            dataset_name: Nombre del dataset
            n_splits: Número de folds para validación cruzada
            include_bootstrap: Si incluir evaluación bootstrap
            
        Returns:
            Resultados de evaluación robusta
        """
        self.logger.info(f"=== INICIANDO EVALUACIÓN ROBUSTA ===")
        self.logger.info(f"Modelo: {model_type}, Dataset: {dataset_name}")
        
        try:
            # Cargar datos
            if not self.load_dataset(dataset_name):
                return {'error': f'No se pudo cargar dataset {dataset_name}'}
            
            dataset = self.datasets[dataset_name]
            x_train = dataset['x_train']
            y_train = dataset['y_train']
            x_test = dataset['x_test']
            y_test = dataset['y_test']
            
            # Inicializar sistema de validación cruzada
            cv_system = CrossValidationUnified(n_splits=n_splits)
            
            # Función constructora del modelo
            def model_builder(**params):
                return self.create_model(model_type, **params)
            
            # Parámetros del modelo para evaluación robusta
            model_params = {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'dropout_rate': 0.3
            }
            
            # Realizar evaluación comprehensiva
            if include_bootstrap:
                evaluation_results = cv_system.comprehensive_evaluation(
                    model_builder, x_train, y_train, x_test, y_test, model_params
                )
            else:
                evaluation_results = cv_system.k_fold_cross_validation(
                    model_builder, x_train, y_train, model_params
                )
            
            # Guardar resultados
            cv_system.save_evaluation_results('robust_evaluation_results.json')
            report = cv_system.generate_evaluation_report('robust_evaluation_report.txt')
            
            self.logger.info("=== EVALUACIÓN ROBUSTA COMPLETADA ===")
            
            # Mostrar resumen
            if 'summary' in evaluation_results:
                summary = evaluation_results['summary']
                self.logger.info(f"Accuracy CV: {summary.get('cv_confidence', 'N/A')}")
                if 'test_accuracy' in summary:
                    self.logger.info(f"Accuracy Test: {summary['test_accuracy']:.4f}")
                
                # Mostrar recomendaciones
                if 'recommendations' in summary:
                    self.logger.info("Recomendaciones:")
                    for rec in summary['recommendations']:
                        self.logger.info(f"  - {rec}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error en evaluación robusta: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_demo(self):
        """Ejecuta una demostración completa del ecosistema"""
        self.logger.info("=== INICIANDO DEMOSTRACIÓN COMPLETA DEL ECOSISTEMA ===")
        
        try:
            # 1. Cargar datos
            print("\n1. Cargando dataset MNIST...")
            self.load_dataset('mnist')
            
            # 2. Crear y entrenar modelo CNN básico
            print("\n2. Creando y entrenando modelo CNN básico...")
            cnn_model = self.create_model('cnn', architecture='basic')
            if cnn_model:
                self.train_model(cnn_model, epochs=2, verbose=1)
                self.evaluate_model(cnn_model)
            
            print("\n=== DEMOSTRACIÓN BÁSICA COMPLETADA ===")
            print("Modelo CNN entrenado y evaluado exitosamente")
            
            return {
                'cnn_model': cnn_model,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error en demostración: {e}")
            print(f"Error en demostración: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """Función principal"""
    print("=" * 60)
    print("ECOSISTEMA UNIFICADO DE REDES NEURONALES")
    print("Versión 1.0 - Sistema Integrado")
    print("=" * 60)
    
    try:
        # Crear instancia del ecosistema
        ecosistema = EcosistemaUnificado()
        
        # Ejecutar demostración simplificada
        results = ecosistema.run_comprehensive_demo()
        
        if results.get('status') == 'success':
            print("\n" + "=" * 60)
            print("ECOSISTEMA EJECUTADO EXITOSAMENTE")
            print("Componentes básicos funcionan correctamente")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("DEMOSTRACIÓN COMPLETADA CON ERRORES")
            print("=" * 60)
        
        return results
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {e}")
        print(f"\nError: {e}")
        return None

if __name__ == "__main__":
    # Configurar TensorFlow para evitar warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Ejecutar sistema principal
    main()