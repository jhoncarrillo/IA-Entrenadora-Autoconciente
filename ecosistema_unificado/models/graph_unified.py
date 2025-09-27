"""
Modelo de Grafos Unificado del Ecosistema de Redes Neuronales
============================================================

Implementación unificada de Graph Neural Networks.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class GraphUnified:
    """Clase unificada para modelos de grafos"""
    
    def __init__(self, num_nodes, node_features, num_classes):
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Construye el modelo GNN"""
        # Entradas
        node_input = keras.Input(shape=(self.num_nodes, self.node_features))
        adjacency_input = keras.Input(shape=(self.num_nodes, self.num_nodes))
        
        # Capas de convolución de grafos
        x = self._graph_conv_layer(node_input, adjacency_input, 64)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        x = self._graph_conv_layer(x, adjacency_input, 32)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
        
        # Pooling global
        x = tf.reduce_mean(x, axis=1)  # Global mean pooling
        
        # Clasificación
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model([node_input, adjacency_input], outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _graph_conv_layer(self, node_features, adjacency_matrix, units):
        """Capa de convolución de grafos"""
        # Transformación lineal de características
        transformed_features = layers.Dense(units, use_bias=False)(node_features)
        
        # Agregación basada en adyacencia
        aggregated = tf.matmul(adjacency_matrix, transformed_features)
        
        return aggregated
    
    def train(self, node_features, adjacency_matrices, labels, epochs=50, batch_size=32):
        """Entrena el modelo GNN"""
        return self.model.fit(
            [node_features, adjacency_matrices], labels,
            epochs=epochs,
            batch_size=batch_size
        )