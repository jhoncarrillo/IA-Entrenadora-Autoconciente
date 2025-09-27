"""
Sistema de Evolución Unificado del Ecosistema de Redes Neuronales
================================================================

Sistema unificado para evolución automática de arquitecturas de redes neuronales.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from typing import List, Dict, Any, Tuple
import copy

class EvolutionUnified:
    """Clase unificada para evolución de redes neuronales"""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 2):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # Configuraciones posibles para evolución
        self.layer_types = ['dense', 'conv2d', 'lstm', 'gru']
        self.activations = ['relu', 'tanh', 'sigmoid', 'swish']
        self.optimizers = ['adam', 'sgd', 'rmsprop']
        
    def initialize_population(self, input_shape: Tuple, num_classes: int):
        """Inicializa la población con arquitecturas aleatorias"""
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.population = []
        
        for _ in range(self.population_size):
            individual = self._create_random_architecture()
            self.population.append(individual)
        
        print(f"Población inicializada con {self.population_size} individuos")
    
    def _create_random_architecture(self) -> Dict[str, Any]:
        """Crea una arquitectura aleatoria"""
        
        architecture = {
            'layers': [],
            'optimizer': random.choice(self.optimizers),
            'learning_rate': random.uniform(0.0001, 0.01),
            'batch_size': random.choice([16, 32, 64, 128])
        }
        
        # Número aleatorio de capas
        num_layers = random.randint(2, 8)
        
        for i in range(num_layers):
            if len(self.input_shape) == 3:  # Datos de imagen
                layer_type = random.choice(['conv2d', 'dense'])
            else:  # Datos secuenciales
                layer_type = random.choice(['dense', 'lstm', 'gru'])
            
            if layer_type == 'dense':
                layer = {
                    'type': 'dense',
                    'units': random.choice([32, 64, 128, 256, 512]),
                    'activation': random.choice(self.activations),
                    'dropout': random.uniform(0.0, 0.5)
                }
            elif layer_type == 'conv2d':
                layer = {
                    'type': 'conv2d',
                    'filters': random.choice([16, 32, 64, 128]),
                    'kernel_size': random.choice([3, 5]),
                    'activation': random.choice(self.activations),
                    'pool_size': random.choice([2, 3])
                }
            elif layer_type in ['lstm', 'gru']:
                layer = {
                    'type': layer_type,
                    'units': random.choice([32, 64, 128, 256]),
                    'return_sequences': i < num_layers - 2,
                    'dropout': random.uniform(0.0, 0.3)
                }
            
            architecture['layers'].append(layer)
        
        return architecture
    
    def build_model_from_architecture(self, architecture: Dict[str, Any]) -> keras.Model:
        """Construye un modelo de Keras desde una arquitectura"""
        
        model = keras.Sequential()
        
        # Capa de entrada
        model.add(keras.Input(shape=self.input_shape))
        
        for i, layer_config in enumerate(architecture['layers']):
            if layer_config['type'] == 'dense':
                model.add(keras.layers.Dense(
                    units=layer_config['units'],
                    activation=layer_config['activation']
                ))
                if 'dropout' in layer_config and layer_config['dropout'] > 0:
                    model.add(keras.layers.Dropout(layer_config['dropout']))
                    
            elif layer_config['type'] == 'conv2d':
                model.add(keras.layers.Conv2D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation'],
                    padding='same'
                ))
                model.add(keras.layers.MaxPooling2D(
                    pool_size=layer_config['pool_size']
                ))
                
            elif layer_config['type'] == 'lstm':
                model.add(keras.layers.LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.0)
                ))
                
            elif layer_config['type'] == 'gru':
                model.add(keras.layers.GRU(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.0)
                ))
        
        # Aplanar si es necesario
        if len(model.layers) > 0 and len(model.output_shape) > 2:
            model.add(keras.layers.Flatten())
        
        # Capa de salida
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.get({
                'class_name': architecture['optimizer'],
                'config': {'learning_rate': architecture['learning_rate']}
            }),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs: int = 5):
        """Evalúa toda la población"""
        
        self.fitness_scores = []
        
        for i, individual in enumerate(self.population):
            try:
                model = self.build_model_from_architecture(individual)
                
                # Entrenar brevemente
                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=individual['batch_size'],
                    verbose=0
                )
                
                # Fitness basado en precisión de validación
                fitness = max(history.history['val_accuracy'])
                self.fitness_scores.append(fitness)
                
                print(f"Individuo {i+1}: Fitness = {fitness:.4f}")
                
            except Exception as e:
                print(f"Error evaluando individuo {i+1}: {e}")
                self.fitness_scores.append(0.0)
    
    def selection(self) -> List[Dict[str, Any]]:
        """Selección por torneo"""
        
        selected = []
        
        # Mantener élite
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            selected.append(copy.deepcopy(self.population[idx]))
        
        # Selección por torneo para el resto
        while len(selected) < self.population_size:
            tournament_size = 3
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(copy.deepcopy(self.population[winner_idx]))
        
        return selected
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Cruzamiento de dos individuos"""
        
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Intercambiar algunas capas
        min_layers = min(len(parent1['layers']), len(parent2['layers']))
        if min_layers > 1:
            crossover_point = random.randint(1, min_layers - 1)
            
            child1['layers'] = parent1['layers'][:crossover_point] + parent2['layers'][crossover_point:]
            child2['layers'] = parent2['layers'][:crossover_point] + parent1['layers'][crossover_point:]
        
        # Intercambiar hiperparámetros
        if random.random() < 0.5:
            child1['optimizer'] = parent2['optimizer']
            child2['optimizer'] = parent1['optimizer']
        
        return child1, child2
    
    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Muta un individuo"""
        
        mutated = copy.deepcopy(individual)
        
        if random.random() < self.mutation_rate:
            # Mutar hiperparámetros
            if random.random() < 0.3:
                mutated['learning_rate'] = random.uniform(0.0001, 0.01)
            
            if random.random() < 0.3:
                mutated['batch_size'] = random.choice([16, 32, 64, 128])
            
            # Mutar capas
            for layer in mutated['layers']:
                if random.random() < 0.2:
                    if layer['type'] == 'dense' and random.random() < 0.5:
                        layer['units'] = random.choice([32, 64, 128, 256, 512])
                    elif layer['type'] == 'conv2d' and random.random() < 0.5:
                        layer['filters'] = random.choice([16, 32, 64, 128])
        
        return mutated
    
    def evolve_generation(self, x_train, y_train, x_val, y_val, epochs: int = 5):
        """Evoluciona una generación"""
        
        print(f"\n=== Generación {self.generation + 1} ===")
        
        # Evaluar población actual
        self.evaluate_population(x_train, y_train, x_val, y_val, epochs)
        
        # Selección
        selected = self.selection()
        
        # Crear nueva población
        new_population = []
        
        # Mantener élite
        for i in range(self.elite_size):
            new_population.append(selected[i])
        
        # Generar descendencia
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Truncar si es necesario
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Mostrar estadísticas
        best_fitness = max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        
        print(f"Mejor fitness: {best_fitness:.4f}")
        print(f"Fitness promedio: {avg_fitness:.4f}")
        
        return best_fitness, avg_fitness
    
    def get_best_individual(self) -> Dict[str, Any]:
        """Obtiene el mejor individuo de la población actual"""
        
        if not self.fitness_scores:
            return None
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]