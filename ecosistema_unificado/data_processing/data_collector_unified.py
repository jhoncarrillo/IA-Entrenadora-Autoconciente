"""
Colector de Datos Unificado del Ecosistema de Redes Neuronales
=============================================================

Sistema unificado para recolección de datos de múltiples fuentes.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from typing import Tuple, Dict, Any, Optional

class DataCollectorUnified:
    """Clase unificada para recolección de datos"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.datasets = {}
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
    
    def collect_mnist_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Recolecta datos MNIST"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalizar
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Expandir dimensiones para CNN
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        self.datasets['mnist'] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        
        return x_train, y_train, x_test, y_test
    
    def collect_cifar10_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Recolecta datos CIFAR-10"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalizar
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Aplanar etiquetas
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        self.datasets['cifar10'] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        
        return x_train, y_train, x_test, y_test
    
    def generate_synthetic_data(self, 
                              num_samples: int = 1000,
                              input_shape: Tuple[int, ...] = (28, 28, 1),
                              num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos para pruebas"""
        x_data = np.random.rand(num_samples, *input_shape).astype('float32')
        y_data = np.random.randint(0, num_classes, num_samples)
        
        dataset_name = f'synthetic_{num_samples}_{input_shape}'
        self.datasets[dataset_name] = {
            'x_data': x_data,
            'y_data': y_data
        }
        
        return x_data, y_data
    
    def generate_sequence_data(self, 
                             num_samples: int = 1000,
                             sequence_length: int = 50,
                             num_features: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos de secuencia para RNN"""
        x_data = np.random.randn(num_samples, sequence_length, num_features).astype('float32')
        y_data = np.random.randint(0, 2, (num_samples, 1))
        
        self.datasets['sequence'] = {
            'x_data': x_data,
            'y_data': y_data
        }
        
        return x_data, y_data
    
    def get_dataset(self, name: str) -> Optional[Dict[str, np.ndarray]]:
        """Obtiene un dataset por nombre"""
        return self.datasets.get(name)
    
    def list_datasets(self) -> list:
        """Lista todos los datasets disponibles"""
        return list(self.datasets.keys())
    
    def save_dataset(self, name: str, filepath: str = None):
        """Guarda un dataset en disco"""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' no encontrado")
        
        if filepath is None:
            filepath = os.path.join(self.data_dir, f"{name}.npz")
        
        dataset = self.datasets[name]
        np.savez_compressed(filepath, **dataset)
        
        print(f"Dataset '{name}' guardado en: {filepath}")
    
    def load_dataset(self, filepath: str, name: str = None) -> str:
        """Carga un dataset desde disco"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        data = np.load(filepath)
        dataset = {key: data[key] for key in data.files}
        
        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]
        
        self.datasets[name] = dataset
        print(f"Dataset cargado como: {name}")
        
        return name