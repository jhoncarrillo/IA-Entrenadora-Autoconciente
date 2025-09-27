"""
Modelo de Atención Unificado del Ecosistema de Redes Neuronales
==============================================================

Implementación unificada de mecanismos de atención.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class AttentionUnified:
    """Clase unificada para modelos con mecanismos de atención"""
    
    def __init__(self, input_shape, num_classes, attention_type='self'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.attention_type = attention_type
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Construye el modelo con atención"""
        inputs = keras.Input(shape=self.input_shape)
        
        if self.attention_type == 'self':
            x = self._self_attention_block(inputs)
        elif self.attention_type == 'multi_head':
            x = self._multi_head_attention_block(inputs)
        else:
            x = self._basic_attention_block(inputs)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _self_attention_block(self, x):
        """Bloque de auto-atención"""
        attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        attention = layers.Dropout(0.1)(attention)
        x = layers.LayerNormalization()(x + attention)
        
        ffn = layers.Dense(128, activation='relu')(x)
        ffn = layers.Dense(x.shape[-1])(ffn)
        ffn = layers.Dropout(0.1)(ffn)
        
        return layers.LayerNormalization()(x + ffn)
    
    def _multi_head_attention_block(self, x):
        """Bloque de atención multi-cabeza"""
        return layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    
    def _basic_attention_block(self, x):
        """Bloque de atención básica"""
        attention_weights = layers.Dense(1, activation='tanh')(x)
        attention_weights = layers.Softmax(axis=1)(attention_weights)
        return layers.Multiply()([x, attention_weights])
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32):
        """Entrena el modelo"""
        validation_data = (x_val, y_val) if x_val is not None else None
        return self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )