"""
Sistema de Auto-Optimización Unificado del Ecosistema de Redes Neuronales
=========================================================================

Sistema unificado para optimización automática de hiperparámetros.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import ParameterGrid
import itertools

class AutoOptimizationUnified:
    """Clase unificada para auto-optimización de hiperparámetros"""
    
    def __init__(self, optimization_method: str = 'random_search'):
        self.optimization_method = optimization_method
        self.best_params = None
        self.best_score = 0.0
        self.optimization_history = []
        
        # Espacios de búsqueda por defecto
        self.default_search_space = {
            'learning_rate': [0.001, 0.01, 0.1, 0.0001],
            'batch_size': [16, 32, 64, 128],
            'epochs': [10, 20, 50],
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'dropout_rate': [0.0, 0.2, 0.3, 0.5],
            'hidden_units': [64, 128, 256, 512],
            'num_layers': [2, 3, 4, 5]
        }
        
        # Espacios de búsqueda para producción (más épocas y configuraciones robustas)
        self.production_search_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.00005],
            'batch_size': [16, 32, 64],
            'epochs': [50, 75, 100, 150, 200],
            'optimizer': ['adam', 'rmsprop'],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5],
            'hidden_units': [128, 256, 512, 1024],
            'num_layers': [3, 4, 5, 6],
            'batch_norm': [True, False],
            'architecture': ['basic', 'advanced', 'resnet_like']
        }
    
    def random_search(self, 
                     model_builder_func,
                     x_train, y_train, 
                     x_val, y_val,
                     search_space: Dict[str, List] = None,
                     n_trials: int = 20) -> Dict[str, Any]:
        """Búsqueda aleatoria de hiperparámetros"""
        
        if search_space is None:
            search_space = self.default_search_space
        
        print(f"Iniciando búsqueda aleatoria con {n_trials} pruebas...")
        
        for trial in range(n_trials):
            # Generar parámetros aleatorios
            params = {}
            for param, values in search_space.items():
                params[param] = random.choice(values)
            
            try:
                # Construir y entrenar modelo
                model = model_builder_func(**params)
                
                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=params.get('epochs', 10),
                    batch_size=params.get('batch_size', 32),
                    verbose=0
                )
                
                # Evaluar rendimiento
                score = max(history.history['val_accuracy'])
                
                # Actualizar mejor resultado
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                self.optimization_history.append({
                    'trial': trial + 1,
                    'params': params,
                    'score': score
                })
                
                print(f"Prueba {trial + 1}/{n_trials}: Score = {score:.4f}")
                
            except Exception as e:
                print(f"Error en prueba {trial + 1}: {e}")
                continue
        
        print(f"\nMejor score: {self.best_score:.4f}")
        print(f"Mejores parámetros: {self.best_params}")
        
        return self.best_params
    
    def grid_search(self,
                   model_builder_func,
                   x_train, y_train,
                   x_val, y_val,
                   search_space: Dict[str, List] = None) -> Dict[str, Any]:
        """Búsqueda en grilla de hiperparámetros"""
        
        if search_space is None:
            # Usar un subconjunto más pequeño para grid search
            search_space = {
                'learning_rate': [0.001, 0.01],
                'batch_size': [32, 64],
                'optimizer': ['adam', 'sgd'],
                'dropout_rate': [0.2, 0.3]
            }
        
        param_grid = ParameterGrid(search_space)
        total_combinations = len(param_grid)
        
        print(f"Iniciando búsqueda en grilla con {total_combinations} combinaciones...")
        
        for i, params in enumerate(param_grid):
            try:
                # Construir y entrenar modelo
                model = model_builder_func(**params)
                
                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=params.get('epochs', 10),
                    batch_size=params.get('batch_size', 32),
                    verbose=0
                )
                
                # Evaluar rendimiento
                score = max(history.history['val_accuracy'])
                
                # Actualizar mejor resultado
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                self.optimization_history.append({
                    'trial': i + 1,
                    'params': params,
                    'score': score
                })
                
                print(f"Combinación {i + 1}/{total_combinations}: Score = {score:.4f}")
                
            except Exception as e:
                print(f"Error en combinación {i + 1}: {e}")
                continue
        
        print(f"\nMejor score: {self.best_score:.4f}")
        print(f"Mejores parámetros: {self.best_params}")
        
        return self.best_params
    
    def bayesian_optimization(self,
                            model_builder_func,
                            x_train, y_train,
                            x_val, y_val,
                            n_trials: int = 15) -> Dict[str, Any]:
        """Optimización bayesiana simplificada"""
        
        # Implementación simplificada usando muestreo inteligente
        print(f"Iniciando optimización bayesiana con {n_trials} pruebas...")
        
        # Parámetros para exploración vs explotación
        exploration_trials = n_trials // 3
        exploitation_trials = n_trials - exploration_trials
        
        # Fase de exploración
        for trial in range(exploration_trials):
            params = self._sample_random_params()
            score = self._evaluate_params(model_builder_func, params, x_train, y_train, x_val, y_val)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            print(f"Exploración {trial + 1}/{exploration_trials}: Score = {score:.4f}")
        
        # Fase de explotación (refinamiento alrededor de los mejores parámetros)
        for trial in range(exploitation_trials):
            params = self._sample_around_best_params()
            score = self._evaluate_params(model_builder_func, params, x_train, y_train, x_val, y_val)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            print(f"Explotación {trial + 1}/{exploitation_trials}: Score = {score:.4f}")
        
        print(f"\nMejor score: {self.best_score:.4f}")
        print(f"Mejores parámetros: {self.best_params}")
        
        return self.best_params
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Muestrea parámetros aleatorios"""
        params = {}
        for param, values in self.default_search_space.items():
            params[param] = random.choice(values)
        return params
    
    def _sample_around_best_params(self) -> Dict[str, Any]:
        """Muestrea parámetros alrededor de los mejores encontrados"""
        if self.best_params is None:
            return self._sample_random_params()
        
        params = self.best_params.copy()
        
        # Perturbar algunos parámetros
        for param in random.sample(list(params.keys()), k=min(2, len(params))):
            if param in self.default_search_space:
                current_value = params[param]
                possible_values = self.default_search_space[param]
                
                # Elegir un valor cercano
                if current_value in possible_values:
                    current_idx = possible_values.index(current_value)
                    # Elegir índices adyacentes
                    adjacent_indices = [
                        max(0, current_idx - 1),
                        min(len(possible_values) - 1, current_idx + 1)
                    ]
                    params[param] = possible_values[random.choice(adjacent_indices)]
                else:
                    params[param] = random.choice(possible_values)
        
        return params
    
    def _evaluate_params(self, model_builder_func, params, x_train, y_train, x_val, y_val) -> float:
        """Evalúa un conjunto de parámetros"""
        try:
            model = model_builder_func(**params)
            
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=params.get('epochs', 10),
                batch_size=params.get('batch_size', 32),
                verbose=0
            )
            
            score = max(history.history['val_accuracy'])
            
            self.optimization_history.append({
                'params': params,
                'score': score
            })
            
            return score
            
        except Exception as e:
            print(f"Error evaluando parámetros: {e}")
            return 0.0
    
    def optimize(self,
                model_builder_func,
                x_train, y_train,
                x_val, y_val,
                search_space: Dict[str, List] = None,
                n_trials: int = 20) -> Dict[str, Any]:
        """Optimiza usando el método especificado"""
        
        if self.optimization_method == 'random_search':
            return self.random_search(model_builder_func, x_train, y_train, x_val, y_val, search_space, n_trials)
        elif self.optimization_method == 'grid_search':
            return self.grid_search(model_builder_func, x_train, y_train, x_val, y_val, search_space)
        elif self.optimization_method == 'bayesian':
            return self.bayesian_optimization(model_builder_func, x_train, y_train, x_val, y_val, n_trials)
        else:
            raise ValueError(f"Método de optimización no soportado: {self.optimization_method}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la optimización"""
        if not self.optimization_history:
            return {"message": "No se ha ejecutado ninguna optimización"}
        
        scores = [entry['score'] for entry in self.optimization_history]
        
        return {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'total_trials': len(self.optimization_history),
            'average_score': np.mean(scores),
            'std_score': np.std(scores),
            'improvement': self.best_score - scores[0] if scores else 0
        }
    
    def production_optimization(self,
                              model_builder_func,
                              x_train, y_train,
                              x_val, y_val,
                              n_trials: int = 30) -> Dict[str, Any]:
        """Optimización específica para producción con configuraciones robustas"""
        
        print(f"Iniciando optimización para producción con {n_trials} pruebas...")
        print("Usando configuraciones robustas con más épocas y parámetros optimizados")
        
        best_score = 0.0
        best_params = None
        trial_results = []
        
        # Configuraciones predefinidas para producción
        production_configs = [
            {
                'learning_rate': 0.0001,
                'batch_size': 32,
                'epochs': 100,
                'optimizer': 'adam',
                'dropout_rate': 0.3,
                'architecture': 'advanced',
                'batch_norm': True
            },
            {
                'learning_rate': 0.0005,
                'batch_size': 64,
                'epochs': 150,
                'optimizer': 'adam',
                'dropout_rate': 0.4,
                'architecture': 'resnet_like',
                'batch_norm': True
            },
            {
                'learning_rate': 0.00005,
                'batch_size': 16,
                'epochs': 200,
                'optimizer': 'rmsprop',
                'dropout_rate': 0.2,
                'architecture': 'advanced',
                'batch_norm': True
            }
        ]
        
        # Probar configuraciones predefinidas primero
        for i, config in enumerate(production_configs):
            print(f"\nProbando configuración de producción {i+1}/{len(production_configs)}")
            print(f"Configuración: {config}")
            
            try:
                score = self._evaluate_params(model_builder_func, config, x_train, y_train, x_val, y_val)
                
                trial_results.append({
                    'trial': i+1,
                    'params': config.copy(),
                    'score': score,
                    'config_type': 'predefined_production'
                })
                
                if score > best_score:
                    best_score = score
                    best_params = config.copy()
                    print(f"¡Nueva mejor configuración! Score: {score:.4f}")
                
            except Exception as e:
                print(f"Error en configuración {i+1}: {e}")
                continue
        
        # Búsqueda aleatoria adicional en el espacio de producción
        remaining_trials = n_trials - len(production_configs)
        if remaining_trials > 0:
            print(f"\nRealizando {remaining_trials} pruebas adicionales con búsqueda aleatoria...")
            
            for trial in range(remaining_trials):
                # Generar parámetros aleatorios del espacio de producción
                params = {}
                for param, values in self.production_search_space.items():
                    params[param] = random.choice(values)
                
                print(f"\nPrueba {trial + len(production_configs) + 1}/{n_trials}")
                print(f"Parámetros: {params}")
                
                try:
                    score = self._evaluate_params(model_builder_func, params, x_train, y_train, x_val, y_val)
                    
                    trial_results.append({
                        'trial': trial + len(production_configs) + 1,
                        'params': params.copy(),
                        'score': score,
                        'config_type': 'random_production'
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        print(f"¡Nueva mejor configuración! Score: {score:.4f}")
                
                except Exception as e:
                    print(f"Error en prueba {trial + len(production_configs) + 1}: {e}")
                    continue
        
        # Actualizar mejores resultados
        self.best_params = best_params
        self.best_score = best_score
        self.optimization_history.extend(trial_results)
        
        print(f"\n=== OPTIMIZACIÓN PARA PRODUCCIÓN COMPLETADA ===")
        print(f"Mejor score obtenido: {best_score:.4f}")
        print(f"Mejores parámetros: {best_params}")
        print(f"Total de pruebas realizadas: {len(trial_results)}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trial_results': trial_results,
            'optimization_type': 'production',
            'total_trials': len(trial_results)
        }