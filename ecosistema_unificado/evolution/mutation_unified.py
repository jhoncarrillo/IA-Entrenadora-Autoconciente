"""
Sistema de Mutación Unificado del Ecosistema de Redes Neuronales
===============================================================

Sistema unificado para mutación de arquitecturas de redes neuronales.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import random
import copy
from typing import Dict, Any, List

class MutationUnified:
    """Clase unificada para mutación de redes neuronales"""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        
        # Opciones de mutación
        self.layer_types = ['dense', 'conv2d', 'lstm', 'gru']
        self.activations = ['relu', 'tanh', 'sigmoid', 'swish']
        self.optimizers = ['adam', 'sgd', 'rmsprop']
    
    def mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Muta una arquitectura completa"""
        mutated = copy.deepcopy(architecture)
        
        # Mutar hiperparámetros globales
        if random.random() < self.mutation_rate:
            mutated = self._mutate_global_params(mutated)
        
        # Mutar capas individuales
        for i, layer in enumerate(mutated['layers']):
            if random.random() < self.mutation_rate:
                mutated['layers'][i] = self._mutate_layer(layer)
        
        # Posibilidad de agregar/eliminar capas
        if random.random() < self.mutation_rate * 0.5:
            mutated = self._mutate_structure(mutated)
        
        return mutated
    
    def _mutate_global_params(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Muta parámetros globales de la arquitectura"""
        mutated = copy.deepcopy(architecture)
        
        # Mutar learning rate
        if random.random() < 0.3:
            current_lr = mutated.get('learning_rate', 0.001)
            factor = random.uniform(0.5, 2.0)
            mutated['learning_rate'] = max(0.0001, min(0.1, current_lr * factor))
        
        # Mutar batch size
        if random.random() < 0.3:
            mutated['batch_size'] = random.choice([16, 32, 64, 128])
        
        # Mutar optimizer
        if random.random() < 0.2:
            mutated['optimizer'] = random.choice(self.optimizers)
        
        return mutated
    
    def _mutate_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Muta una capa individual"""
        mutated_layer = copy.deepcopy(layer)
        
        if layer['type'] == 'dense':
            mutated_layer = self._mutate_dense_layer(mutated_layer)
        elif layer['type'] == 'conv2d':
            mutated_layer = self._mutate_conv_layer(mutated_layer)
        elif layer['type'] in ['lstm', 'gru']:
            mutated_layer = self._mutate_rnn_layer(mutated_layer)
        
        return mutated_layer
    
    def _mutate_dense_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Muta una capa densa"""
        mutated = copy.deepcopy(layer)
        
        # Mutar número de unidades
        if random.random() < 0.5:
            current_units = mutated.get('units', 128)
            possible_units = [32, 64, 128, 256, 512]
            # Elegir una opción cercana o aleatoria
            if random.random() < 0.7:  # Mutación conservadora
                if current_units in possible_units:
                    idx = possible_units.index(current_units)
                    new_idx = max(0, min(len(possible_units)-1, idx + random.choice([-1, 1])))
                    mutated['units'] = possible_units[new_idx]
                else:
                    mutated['units'] = random.choice(possible_units)
            else:  # Mutación radical
                mutated['units'] = random.choice(possible_units)
        
        # Mutar activación
        if random.random() < 0.3:
            mutated['activation'] = random.choice(self.activations)
        
        # Mutar dropout
        if random.random() < 0.3:
            mutated['dropout'] = random.uniform(0.0, 0.5)
        
        return mutated
    
    def _mutate_conv_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Muta una capa convolucional"""
        mutated = copy.deepcopy(layer)
        
        # Mutar número de filtros
        if random.random() < 0.5:
            current_filters = mutated.get('filters', 64)
            possible_filters = [16, 32, 64, 128, 256]
            if current_filters in possible_filters:
                idx = possible_filters.index(current_filters)
                new_idx = max(0, min(len(possible_filters)-1, idx + random.choice([-1, 1])))
                mutated['filters'] = possible_filters[new_idx]
            else:
                mutated['filters'] = random.choice(possible_filters)
        
        # Mutar kernel size
        if random.random() < 0.3:
            mutated['kernel_size'] = random.choice([3, 5, 7])
        
        # Mutar activación
        if random.random() < 0.3:
            mutated['activation'] = random.choice(self.activations)
        
        # Mutar pool size
        if random.random() < 0.2:
            mutated['pool_size'] = random.choice([2, 3])
        
        return mutated
    
    def _mutate_rnn_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Muta una capa RNN"""
        mutated = copy.deepcopy(layer)
        
        # Mutar número de unidades
        if random.random() < 0.5:
            current_units = mutated.get('units', 128)
            possible_units = [32, 64, 128, 256]
            if current_units in possible_units:
                idx = possible_units.index(current_units)
                new_idx = max(0, min(len(possible_units)-1, idx + random.choice([-1, 1])))
                mutated['units'] = possible_units[new_idx]
            else:
                mutated['units'] = random.choice(possible_units)
        
        # Mutar dropout
        if random.random() < 0.3:
            mutated['dropout'] = random.uniform(0.0, 0.3)
        
        # Mutar return_sequences (solo si no es la última capa)
        if random.random() < 0.2:
            mutated['return_sequences'] = not mutated.get('return_sequences', False)
        
        return mutated
    
    def _mutate_structure(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Muta la estructura de la arquitectura (agregar/eliminar capas)"""
        mutated = copy.deepcopy(architecture)
        
        if random.random() < 0.5 and len(mutated['layers']) < 8:
            # Agregar capa
            new_layer = self._create_random_layer()
            insert_position = random.randint(0, len(mutated['layers']))
            mutated['layers'].insert(insert_position, new_layer)
        
        elif len(mutated['layers']) > 2:
            # Eliminar capa
            remove_position = random.randint(0, len(mutated['layers']) - 1)
            mutated['layers'].pop(remove_position)
        
        return mutated
    
    def _create_random_layer(self) -> Dict[str, Any]:
        """Crea una capa aleatoria"""
        layer_type = random.choice(self.layer_types)
        
        if layer_type == 'dense':
            return {
                'type': 'dense',
                'units': random.choice([32, 64, 128, 256]),
                'activation': random.choice(self.activations),
                'dropout': random.uniform(0.0, 0.3)
            }
        elif layer_type == 'conv2d':
            return {
                'type': 'conv2d',
                'filters': random.choice([16, 32, 64, 128]),
                'kernel_size': random.choice([3, 5]),
                'activation': random.choice(self.activations),
                'pool_size': random.choice([2, 3])
            }
        elif layer_type in ['lstm', 'gru']:
            return {
                'type': layer_type,
                'units': random.choice([32, 64, 128]),
                'return_sequences': random.choice([True, False]),
                'dropout': random.uniform(0.0, 0.2)
            }
    
    def crossover_architectures(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruzamiento entre dos arquitecturas"""
        child = copy.deepcopy(parent1)
        
        # Intercambiar algunas capas
        min_layers = min(len(parent1['layers']), len(parent2['layers']))
        if min_layers > 1:
            crossover_point = random.randint(1, min_layers - 1)
            child['layers'] = parent1['layers'][:crossover_point] + parent2['layers'][crossover_point:]
        
        # Intercambiar algunos hiperparámetros
        if random.random() < 0.5:
            child['optimizer'] = parent2['optimizer']
        if random.random() < 0.5:
            child['learning_rate'] = parent2['learning_rate']
        if random.random() < 0.5:
            child['batch_size'] = parent2['batch_size']
        
        return child