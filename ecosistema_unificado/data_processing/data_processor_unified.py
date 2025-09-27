"""
Procesador de Datos Unificado del Ecosistema de Redes Neuronales
===============================================================

Sistema unificado para procesamiento y preparación de datos.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union

class DataProcessorUnified:
    """Clase unificada para procesamiento de datos"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def normalize_data(self, 
                      data: np.ndarray, 
                      method: str = 'minmax',
                      fit_scaler: bool = True) -> np.ndarray:
        """Normaliza los datos usando diferentes métodos"""
        
        original_shape = data.shape
        data_reshaped = data.reshape(data.shape[0], -1)
        
        if method == 'minmax':
            if fit_scaler or 'minmax' not in self.scalers:
                self.scalers['minmax'] = MinMaxScaler()
                normalized = self.scalers['minmax'].fit_transform(data_reshaped)
            else:
                normalized = self.scalers['minmax'].transform(data_reshaped)
                
        elif method == 'standard':
            if fit_scaler or 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                normalized = self.scalers['standard'].fit_transform(data_reshaped)
            else:
                normalized = self.scalers['standard'].transform(data_reshaped)
                
        elif method == 'simple':
            # Normalización simple (0-1)
            normalized = data_reshaped / 255.0 if data_reshaped.max() > 1 else data_reshaped
            
        else:
            raise ValueError(f"Método de normalización no soportado: {method}")
        
        return normalized.reshape(original_shape)
    
    def encode_labels(self, 
                     labels: np.ndarray, 
                     method: str = 'categorical') -> np.ndarray:
        """Codifica las etiquetas"""
        
        if method == 'categorical':
            # One-hot encoding
            num_classes = len(np.unique(labels))
            return tf.keras.utils.to_categorical(labels, num_classes)
            
        elif method == 'label':
            # Label encoding
            if 'label' not in self.encoders:
                self.encoders['label'] = LabelEncoder()
                return self.encoders['label'].fit_transform(labels)
            else:
                return self.encoders['label'].transform(labels)
                
        else:
            return labels
    
    def augment_images(self, 
                      images: np.ndarray, 
                      augmentation_factor: int = 2) -> np.ndarray:
        """Aumenta el dataset de imágenes"""
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        augmented_images = []
        
        for i in range(len(images)):
            img = images[i:i+1]
            
            # Generar imágenes aumentadas
            aug_iter = datagen.flow(img, batch_size=1)
            for j in range(augmentation_factor):
                augmented_images.append(next(aug_iter)[0])
        
        return np.array(augmented_images)
    
    def split_data(self, 
                  x: np.ndarray, 
                  y: np.ndarray, 
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Divide los datos en entrenamiento, validación y prueba"""
        
        # Primera división: entrenamiento + validación vs prueba
        x_temp, x_test, y_temp, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Segunda división: entrenamiento vs validación
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            x_train, x_val, y_train, y_val = train_test_split(
                x_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_temp
            )
            return x_train, x_val, x_test, y_train, y_val, y_test
        else:
            return x_temp, x_test, y_temp, y_test
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        sequence_length: int,
                        step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Crea secuencias para modelos RNN"""
        
        sequences = []
        targets = []
        
        for i in range(0, len(data) - sequence_length, step):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def balance_dataset(self, 
                       x: np.ndarray, 
                       y: np.ndarray, 
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """Balancea el dataset"""
        
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if method == 'undersample':
            # Submuestreo a la clase minoritaria
            min_count = min(counts)
            
            balanced_x = []
            balanced_y = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, min_count, replace=False)
                
                balanced_x.append(x[selected_indices])
                balanced_y.append(y[selected_indices])
            
            return np.vstack(balanced_x), np.hstack(balanced_y)
            
        elif method == 'oversample':
            # Sobremuestreo a la clase mayoritaria
            max_count = max(counts)
            
            balanced_x = []
            balanced_y = []
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, max_count, replace=True)
                
                balanced_x.append(x[selected_indices])
                balanced_y.append(y[selected_indices])
            
            return np.vstack(balanced_x), np.hstack(balanced_y)
        
        else:
            return x, y
    
    def get_data_info(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Obtiene información sobre el dataset"""
        
        unique_classes, counts = np.unique(y, return_counts=True)
        
        info = {
            'samples': len(x),
            'input_shape': x.shape[1:],
            'num_classes': len(unique_classes),
            'class_distribution': dict(zip(unique_classes, counts)),
            'data_type': x.dtype,
            'data_range': (x.min(), x.max())
        }
        
        return info