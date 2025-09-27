"""
Modelo RNN Unificado del Ecosistema de Redes Neuronales
======================================================

Implementación unificada de Recurrent Neural Networks que incluye
LSTM, GRU, y RNN básicas con funcionalidades avanzadas.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import numpy as np
from typing import Tuple, Optional, Dict, Any

class RNNUnified:
    """Clase unificada para modelos RNN con funcionalidades avanzadas"""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 output_dim: int,
                 rnn_type: str = 'lstm',
                 units: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 bidirectional: bool = False,
                 return_sequences: bool = False):
        """
        Inicializa el modelo RNN unificado
        
        Args:
            input_shape: Forma de entrada (timesteps, features)
            output_dim: Dimensión de salida
            rnn_type: Tipo de RNN ('lstm', 'gru', 'simple_rnn')
            units: Número de unidades por capa
            num_layers: Número de capas RNN
            dropout_rate: Tasa de dropout
            bidirectional: Si usar capas bidireccionales
            return_sequences: Si retornar secuencias completas
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.units = units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        
        self.model = None
        self.history = None
        
        self._build_model()
    
    def _build_model(self):
        """Construye el modelo RNN"""
        model = models.Sequential()
        
        # Capas RNN
        for i in range(self.num_layers):
            return_seq = self.return_sequences if i == self.num_layers - 1 else True
            
            if self.rnn_type == 'lstm':
                rnn_layer = layers.LSTM(
                    self.units,
                    return_sequences=return_seq,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            elif self.rnn_type == 'gru':
                rnn_layer = layers.GRU(
                    self.units,
                    return_sequences=return_seq,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            elif self.rnn_type == 'simple_rnn':
                rnn_layer = layers.SimpleRNN(
                    self.units,
                    return_sequences=return_seq,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            else:
                raise ValueError(f"Tipo de RNN no soportado: {self.rnn_type}")
            
            if self.bidirectional:
                rnn_layer = layers.Bidirectional(rnn_layer)
            
            if i == 0:
                model.add(layers.Input(shape=self.input_shape))
            
            model.add(rnn_layer)
            
            if i < self.num_layers - 1:
                model.add(layers.Dropout(self.dropout_rate))
        
        # Capa de salida
        if not self.return_sequences:
            model.add(layers.Dense(self.output_dim, activation='softmax'))
        else:
            model.add(layers.TimeDistributed(layers.Dense(self.output_dim, activation='softmax')))
        
        # Compilar modelo
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=32):
        """Entrena el modelo RNN"""
        validation_data = (x_val, y_val) if x_val is not None else None
        
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, x):
        """Realiza predicciones"""
        return self.model.predict(x)
    
    def evaluate(self, x_test, y_test):
        """Evalúa el modelo"""
        return self.model.evaluate(x_test, y_test)