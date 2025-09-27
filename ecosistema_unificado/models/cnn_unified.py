"""
Modelo CNN Unificado del Ecosistema de Redes Neuronales
======================================================

Implementación unificada de Convolutional Neural Networks que combina
las mejores características de ambos proyectos con funcionalidades avanzadas.

Características:
- Arquitecturas CNN básicas y avanzadas
- Regularización automática
- Optimización adaptativa
- Métricas avanzadas
- Soporte para diferentes tipos de datos

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TerminateOnNaN
import numpy as np
from typing import Tuple, Optional, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_checkpoint_unified import ModelCheckpointUnified

class CNNUnified:
    """
    Clase unificada para modelos CNN con funcionalidades avanzadas
    """
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, 
                 architecture='basic', **kwargs):
        """
        Inicializa el modelo CNN unificado
        
        Args:
            input_shape: Forma de entrada de los datos
            num_classes: Número de clases para clasificación
            architecture: Tipo de arquitectura ('basic', 'advanced', 'resnet')
            **kwargs: Argumentos adicionales
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.model = None
        self.history = None
        
        # Sistema de checkpoints automáticos
        self.checkpoint_system = ModelCheckpointUnified(
            base_path='saved_models/cnn_models',
            auto_save=True,
            keep_best_only=True,
            max_checkpoints=10
        )
        
        # Configuración avanzada
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.l2_reg = kwargs.get('l2_reg', 0.001)
        self.batch_norm = kwargs.get('batch_norm', True)
        
        self._build_model()
    
    def _build_model(self):
        """Construye el modelo según la arquitectura especificada"""
        if self.architecture == 'basic':
            self.model = self._build_basic_cnn()
        elif self.architecture == 'advanced':
            self.model = self._build_advanced_cnn()
        elif self.architecture == 'resnet_like':
            self.model = self._build_resnet_like()
        else:
            raise ValueError(f"Arquitectura no soportada: {self.architecture}")
    
    def _build_basic_cnn(self):
        """Construye una CNN básica"""
        inputs = layers.Input(shape=self.input_shape)
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def _build_advanced_cnn(self):
        """Construye una CNN avanzada con más capas y regularización"""
        model = models.Sequential([
            # Primer bloque
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization() if self.batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Segundo bloque
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization() if self.batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Tercer bloque
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization() if self.batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.dropout_rate * 0.5),
            
            # Clasificador
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization() if self.batch_norm else layers.Lambda(lambda x: x),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_resnet_like(self):
        """Construye una CNN con conexiones residuales"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Capa inicial
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Bloques residuales
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128)
        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256)
        
        # Clasificador
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def _residual_block(self, x, filters, stride=1):
        """Bloque residual para arquitectura tipo ResNet"""
        shortcut = x
        
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x) if self.batch_norm else x
        
        # Ajustar shortcut si es necesario
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut) if self.batch_norm else shortcut
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 10,
              batch_size: int = 32,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Entrena el modelo CNN
        
        Args:
            x_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            x_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño del batch
            verbose: Nivel de verbosidad
            
        Returns:
            Diccionario con el historial de entrenamiento
        """
        try:
            # Validación de datos
            if self.model is None:
                raise ValueError("El modelo no ha sido construido")
            
            # Asegurar que las etiquetas sean enteros
            y_train = y_train.astype(np.int32)
            if y_val is not None:
                y_val = y_val.astype(np.int32)
            
            # Callbacks avanzados para entrenamiento robusto
            callback_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss' if x_val is not None else 'loss',
                    patience=15,  # Más paciencia para entrenamiento robusto
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if x_val is not None else 'loss',
                    factor=0.5,  # Reducción más gradual
                    patience=8,  # Más paciencia antes de reducir LR
                    min_lr=1e-8,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    filepath='best_model_checkpoint.h5',
                    monitor='val_loss' if x_val is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ),
                callbacks.CSVLogger(
                    filename='training_log.csv',
                    append=True
                ),
                callbacks.TerminateOnNaN(),
                # Checkpoint automático personalizado
                self.checkpoint_system.get_checkpoint_callback(
                    model_name=f"cnn_robust_{epochs}epochs",
                    monitor='val_accuracy' if x_val is not None else 'accuracy',
                    mode='max'
                )
            ]
            
            # Datos de validación
            validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
            
            # Entrenamiento
            self.history = self.model.fit(
                x_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callback_list,
                verbose=verbose
            )
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            return {"error": str(e)}
        
        # Guardar modelo final automáticamente
        final_accuracy = self.history.history.get('val_accuracy', self.history.history.get('accuracy', [0]))[-1]
        final_metrics = {
            'accuracy': self.history.history.get('accuracy', [0])[-1],
            'loss': self.history.history.get('loss', [0])[-1],
            'val_accuracy': self.history.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in self.history.history else None,
            'val_loss': self.history.history.get('val_loss', [0])[-1] if 'val_loss' in self.history.history else None,
            'epochs_trained': epochs
        }
        
        # Guardar como mejor modelo si supera el threshold
        model_type = 'best' if final_accuracy > 0.9 else 'checkpoint'
        saved_path = self.checkpoint_system.save_model(
            model=self.model,
            model_name=f"cnn_final_{epochs}epochs",
            score=final_accuracy,
            metrics=final_metrics,
            model_type=model_type,
            metadata={'training_completed': True, 'robust_training': True}
        )
        
        print(f"Modelo final guardado en: {saved_path}")
        
        result = self.history.history.copy()
        result.update({
            'saved_model_path': saved_path,
            'checkpoint_summary': self.checkpoint_system.get_model_summary()
        })
        
        return result
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo
        
        Args:
            x_test: Datos de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            if self.model is None:
                raise ValueError("El modelo no ha sido entrenado")
            
            # Asegurar que las etiquetas sean enteros
            y_test = y_test.astype(np.int32)
            
            results = self.model.evaluate(x_test, y_test, verbose=0)
            metric_names = self.model.metrics_names
            
            return dict(zip(metric_names, results))
            
        except Exception as e:
            print(f"Error durante la evaluación: {e}")
            return {"error": str(e)}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        
        Args:
            x: Datos de entrada
            
        Returns:
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        return self.model.predict(x)
    
    def summary(self):
        """Muestra un resumen del modelo (compatibilidad con Keras)"""
        if self.model is None:
            print("Modelo no construido")
            return
        
        self.model.summary()
    
    def get_model_summary(self) -> str:
        """Retorna un resumen del modelo"""
        if self.model is None:
            return "Modelo no construido"
        
        import io
        import sys
        
        # Capturar el summary en un string
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def save_model(self, filepath: str):
        """Guarda el modelo"""
        if self.model is None:
            raise ValueError("El modelo no ha sido construido")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Carga un modelo guardado"""
        self.model = keras.models.load_model(filepath)