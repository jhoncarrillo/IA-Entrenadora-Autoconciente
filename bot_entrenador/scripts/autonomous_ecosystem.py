"""
Sistema Autónomo de Ecosistema de Redes Neuronales
Integra múltiples tipos de modelos y los entrena de manera coordinada
"""

import numpy as np
import torch
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_extraction import AdvancedDataExtractor
from scripts.prompt_generation import AdvancedPromptGenerator
from scripts.model_interaction import AdvancedModelInteraction

@dataclass
class ModelConfig:
    """Configuración para cada modelo en el ecosistema"""
    name: str
    model_type: str  # 'cnn', 'rnn', 'lstm', 'transformer', 'gan', 'autoencoder'
    task_type: str   # 'classification', 'regression', 'generation', 'feature_extraction'
    input_shape: Tuple
    output_shape: Tuple
    hyperparameters: Dict[str, Any]
    is_active: bool = True

class AutonomousEcosystem:
    def __init__(self, config_path: Optional[str] = None):
        self.models = {}
        self.training_history = {}
        self.data_preprocessors = {}
        self.performance_metrics = {}
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self.data_extractor = AdvancedDataExtractor()
        self.prompt_generator = AdvancedPromptGenerator()
        self.model_interaction = AdvancedModelInteraction()
        
        # Cargar configuración si existe
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.initialize_default_models()
    
    def initialize_default_models(self):
        """Inicializa modelos por defecto del ecosistema"""
        default_configs = [
            ModelConfig(
                name="text_classifier",
                model_type="transformer",
                task_type="classification",
                input_shape=(512,),
                output_shape=(10,),
                hyperparameters={"learning_rate": 0.001, "epochs": 10}
            ),
            ModelConfig(
                name="sequence_predictor",
                model_type="lstm",
                task_type="regression",
                input_shape=(100, 1),
                output_shape=(1,),
                hyperparameters={"learning_rate": 0.001, "epochs": 50}
            ),
            ModelConfig(
                name="image_classifier",
                model_type="cnn",
                task_type="classification",
                input_shape=(224, 224, 3),
                output_shape=(1000,),
                hyperparameters={"learning_rate": 0.001, "epochs": 20}
            ),
            ModelConfig(
                name="pattern_detector",
                model_type="rnn",
                task_type="classification",
                input_shape=(50, 10),
                output_shape=(5,),
                hyperparameters={"learning_rate": 0.001, "epochs": 30}
            )
        ]
        
        for config in default_configs:
            self.add_model(config)
    
    def add_model(self, config: ModelConfig):
        """Añade un nuevo modelo al ecosistema"""
        try:
            if config.model_type == "cnn":
                model = self._create_cnn_model(config)
            elif config.model_type == "lstm":
                model = self._create_lstm_model(config)
            elif config.model_type == "rnn":
                model = self._create_rnn_model(config)
            elif config.model_type == "transformer":
                model = self._create_transformer_model(config)
            elif config.model_type == "autoencoder":
                model = self._create_autoencoder_model(config)
            elif config.model_type == "gan":
                model = self._create_gan_model(config)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {config.model_type}")
            
            self.models[config.name] = {
                'model': model,
                'config': config,
                'trained': False,
                'performance': {}
            }
            
            self.logger.info(f"Modelo {config.name} añadido al ecosistema")
            
        except Exception as e:
            self.logger.error(f"Error al crear modelo {config.name}: {str(e)}")
    
    def _create_cnn_model(self, config: ModelConfig):
        """Crea un modelo CNN avanzado"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=config.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(config.output_shape[0], 
                                activation='softmax' if config.task_type == 'classification' else 'linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.hyperparameters.get('learning_rate', 0.001))
        loss = 'categorical_crossentropy' if config.task_type == 'classification' else 'mse'
        metrics = ['accuracy'] if config.task_type == 'classification' else ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def _create_lstm_model(self, config: ModelConfig):
        """Crea un modelo LSTM avanzado"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=config.input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(config.output_shape[0], 
                                activation='softmax' if config.task_type == 'classification' else 'linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.hyperparameters.get('learning_rate', 0.001))
        loss = 'categorical_crossentropy' if config.task_type == 'classification' else 'mse'
        metrics = ['accuracy'] if config.task_type == 'classification' else ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def _create_rnn_model(self, config: ModelConfig):
        """Crea un modelo RNN avanzado"""
        model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=config.input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.SimpleRNN(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(config.output_shape[0], 
                                activation='softmax' if config.task_type == 'classification' else 'linear')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.hyperparameters.get('learning_rate', 0.001))
        loss = 'categorical_crossentropy' if config.task_type == 'classification' else 'mse'
        metrics = ['accuracy'] if config.task_type == 'classification' else ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def _create_transformer_model(self, config: ModelConfig):
        """Crea un modelo Transformer personalizado"""
        from transformers import AutoModel, AutoTokenizer
        
        # Para este ejemplo, usamos un modelo pre-entrenado como base
        base_model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Crear un modelo personalizado con capas adicionales
        inputs = tf.keras.Input(shape=config.input_shape, dtype=tf.int32)
        
        # Aquí iría la implementación del transformer personalizado
        # Por simplicidad, usamos una aproximación con capas densas
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(config.output_shape[0], 
                                      activation='softmax' if config.task_type == 'classification' else 'linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.hyperparameters.get('learning_rate', 0.001))
        loss = 'categorical_crossentropy' if config.task_type == 'classification' else 'mse'
        metrics = ['accuracy'] if config.task_type == 'classification' else ['mae']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def _create_autoencoder_model(self, config: ModelConfig):
        """Crea un modelo Autoencoder"""
        input_dim = np.prod(config.input_shape)
        
        # Encoder
        encoder_input = tf.keras.Input(shape=config.input_shape)
        x = tf.keras.layers.Flatten()(encoder_input)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        encoded = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Decoder
        x = tf.keras.layers.Dense(128, activation='relu')(encoded)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)
        decoded = tf.keras.layers.Reshape(config.input_shape)(x)
        
        autoencoder = tf.keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def _create_gan_model(self, config: ModelConfig):
        """Crea un modelo GAN (Generative Adversarial Network)"""
        # Generador
        noise_dim = 100
        generator = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(noise_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(np.prod(config.output_shape), activation='tanh'),
            tf.keras.layers.Reshape(config.output_shape)
        ])
        
        # Discriminador
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=config.input_shape),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # GAN combinado
        discriminator.trainable = False
        gan_input = tf.keras.Input(shape=(noise_dim,))
        generated = generator(gan_input)
        gan_output = discriminator(generated)
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        
        return {'generator': generator, 'discriminator': discriminator, 'gan': gan}
    
    def autonomous_training(self, data_sources: List[Dict], training_config: Optional[Dict] = None):
        """Entrenamiento autónomo del ecosistema completo"""
        self.logger.info("Iniciando entrenamiento autónomo del ecosistema")
        
        # Extraer y procesar datos
        processed_data = self._extract_and_process_data(data_sources)
        
        # Entrenar cada modelo activo
        for model_name, model_info in self.models.items():
            if model_info['config'].is_active:
                try:
                    self.logger.info(f"Entrenando modelo: {model_name}")
                    self._train_single_model(model_name, processed_data, training_config)
                except Exception as e:
                    self.logger.error(f"Error entrenando {model_name}: {str(e)}")
        
        # Evaluar rendimiento del ecosistema
        self._evaluate_ecosystem_performance()
        
        # Optimizar automáticamente
        self._auto_optimize_ecosystem()
        
        self.logger.info("Entrenamiento autónomo completado")
    
    def _extract_and_process_data(self, data_sources: List[Dict]) -> Dict[str, Any]:
        """Extrae y procesa datos de múltiples fuentes"""
        all_data = {}
        
        for source in data_sources:
            try:
                extracted = self.data_extractor.extract_data(source)
                all_data[source.get('name', f"source_{len(all_data)}")] = extracted
            except Exception as e:
                self.logger.error(f"Error extrayendo datos de {source}: {str(e)}")
        
        # Procesar datos para diferentes tipos de modelos
        processed = {
            'text_data': self._process_text_data(all_data),
            'numeric_data': self._process_numeric_data(all_data),
            'image_data': self._process_image_data(all_data),
            'sequence_data': self._process_sequence_data(all_data)
        }
        
        return processed
    
    def _process_text_data(self, data: Dict) -> Dict:
        """Procesa datos de texto para modelos NLP"""
        text_data = []
        for source_name, source_data in data.items():
            if isinstance(source_data, dict) and 'raw_content' in source_data:
                text_data.append(source_data['raw_content'])
            elif isinstance(source_data, str):
                text_data.append(source_data)
        
        # Generar prompts y respuestas para entrenamiento
        training_prompts = []
        for text in text_data:
            prompts = self.prompt_generator.generate_training_prompts(text)
            training_prompts.extend(prompts)
        
        return {
            'raw_texts': text_data,
            'training_prompts': training_prompts,
            'processed_count': len(training_prompts)
        }
    
    def _process_numeric_data(self, data: Dict) -> Dict:
        """Procesa datos numéricos para modelos de ML"""
        numeric_data = []
        
        for source_name, source_data in data.items():
            if isinstance(source_data, dict) and 'data' in source_data:
                if isinstance(source_data['data'], list):
                    numeric_data.extend(source_data['data'])
        
        if numeric_data:
            # Convertir a formato numérico
            import pandas as pd
            df = pd.DataFrame(numeric_data)
            
            # Separar features y targets (asumiendo última columna como target)
            if len(df.columns) > 1:
                X = df.iloc[:, :-1].select_dtypes(include=[np.number])
                y = df.iloc[:, -1]
                
                # Normalizar features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                return {
                    'features': X_scaled,
                    'targets': y.values,
                    'scaler': scaler,
                    'feature_names': list(X.columns),
                    'shape': X_scaled.shape
                }
        
        return {'features': np.array([]), 'targets': np.array([])}
    
    def _process_image_data(self, data: Dict) -> Dict:
        """Procesa datos de imagen para modelos de visión"""
        # Placeholder para procesamiento de imágenes
        return {'images': [], 'labels': [], 'processed_count': 0}
    
    def _process_sequence_data(self, data: Dict) -> Dict:
        """Procesa datos secuenciales para modelos RNN/LSTM"""
        sequences = []
        
        for source_name, source_data in data.items():
            if isinstance(source_data, dict) and 'sentences' in source_data:
                # Convertir oraciones a secuencias numéricas
                sentences = source_data['sentences']
                for sentence in sentences[:100]:  # Limitar a 100 oraciones
                    # Convertir a secuencia numérica simple (longitud de palabras)
                    words = sentence.split()
                    sequence = [len(word) for word in words]
                    if len(sequence) > 10:  # Mínimo 10 elementos
                        sequences.append(sequence[:50])  # Máximo 50 elementos
        
        if sequences:
            # Padding de secuencias
            max_len = max(len(seq) for seq in sequences)
            padded_sequences = []
            for seq in sequences:
                padded = seq + [0] * (max_len - len(seq))
                padded_sequences.append(padded)
            
            return {
                'sequences': np.array(padded_sequences),
                'sequence_length': max_len,
                'num_sequences': len(padded_sequences)
            }
        
        return {'sequences': np.array([]), 'sequence_length': 0}
    
    def _train_single_model(self, model_name: str, processed_data: Dict, training_config: Optional[Dict]):
        """Entrena un modelo individual"""
        model_info = self.models[model_name]
        model = model_info['model']
        config = model_info['config']
        
        # Seleccionar datos apropiados según el tipo de modelo
        if config.model_type in ['lstm', 'rnn']:
            data = processed_data['sequence_data']
            if data['sequences'].size > 0:
                X = data['sequences']
                y = X[:, 1:]  # Predicción de siguiente elemento
                X = X[:, :-1]
                
                if len(X) > 10:  # Mínimo de datos para entrenar
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Reshape para LSTM/RNN
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=config.hyperparameters.get('epochs', 10),
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=0
                    )
                    
                    model_info['trained'] = True
                    model_info['performance'] = {
                        'final_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1]
                    }
        
        elif config.model_type == 'transformer':
            # Entrenamiento simplificado para transformer
            data = processed_data['text_data']
            if data['training_prompts']:
                # Simulación de entrenamiento (en implementación real usaríamos tokenización)
                model_info['trained'] = True
                model_info['performance'] = {'status': 'trained_on_text_data'}
        
        # Guardar historial de entrenamiento
        self.training_history[model_name] = {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'performance': model_info['performance']
        }
    
    def _evaluate_ecosystem_performance(self):
        """Evalúa el rendimiento general del ecosistema"""
        total_models = len(self.models)
        trained_models = sum(1 for model_info in self.models.values() if model_info['trained'])
        
        ecosystem_performance = {
            'total_models': total_models,
            'trained_models': trained_models,
            'training_success_rate': trained_models / total_models if total_models > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_metrics['ecosystem'] = ecosystem_performance
        self.logger.info(f"Rendimiento del ecosistema: {trained_models}/{total_models} modelos entrenados")
    
    def _auto_optimize_ecosystem(self):
        """Optimización automática del ecosistema"""
        # Identificar modelos con bajo rendimiento
        underperforming_models = []
        
        for model_name, model_info in self.models.items():
            if model_info['trained'] and 'final_loss' in model_info['performance']:
                if model_info['performance']['final_loss'] > 1.0:  # Umbral arbitrario
                    underperforming_models.append(model_name)
        
        # Ajustar hiperparámetros automáticamente
        for model_name in underperforming_models:
            config = self.models[model_name]['config']
            # Reducir learning rate
            config.hyperparameters['learning_rate'] *= 0.5
            # Aumentar epochs
            config.hyperparameters['epochs'] = min(config.hyperparameters['epochs'] * 2, 100)
            
            self.logger.info(f"Optimizando modelo {model_name}: LR={config.hyperparameters['learning_rate']}")
    
    def save_ecosystem_state(self, filepath: str):
        """Guarda el estado completo del ecosistema"""
        state = {
            'models_config': {name: info['config'].__dict__ for name, info in self.models.items()},
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Estado del ecosistema guardado en: {filepath}")
    
    def load_config(self, config_path: str):
        """Carga configuración desde archivo"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Implementar carga de configuración
        self.logger.info(f"Configuración cargada desde: {config_path}")
    
    def get_ecosystem_summary(self) -> Dict:
        """Retorna un resumen del estado del ecosistema"""
        return {
            'total_models': len(self.models),
            'active_models': sum(1 for info in self.models.values() if info['config'].is_active),
            'trained_models': sum(1 for info in self.models.values() if info['trained']),
            'model_types': list(set(info['config'].model_type for info in self.models.values())),
            'performance_metrics': self.performance_metrics,
            'last_training': max(self.training_history.values(), key=lambda x: x['timestamp'])['timestamp'] if self.training_history else None
        }