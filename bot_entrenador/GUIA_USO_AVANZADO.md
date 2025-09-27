# 🚀 Guía de Uso Avanzado - Bot Entrenador General

## 📖 Introducción

Esta guía te ayudará a aprovechar al máximo las capacidades avanzadas del **Bot Entrenador General**, un sistema autónomo de entrenamiento de ecosistemas de IA.

## 🏁 Inicio Rápido

### 1. **Instalación de Dependencias**
```bash
pip install -r requirements.txt
```

### 2. **Ejecución Básica**
```bash
python main.py
```

### 3. **Estructura de Datos Recomendada**
```
proyecto/
├── data/
│   ├── textos/          # Archivos .txt
│   ├── estructurados/   # .csv, .json
│   ├── documentos/      # .pdf
│   ├── imagenes/        # .jpg, .png
│   └── bases_datos/     # .db, .sqlite
├── models/              # Modelos entrenados
└── output/              # Resultados y reportes
```

## 🔧 Configuración Avanzada

### 1. **Configuración Personalizada**
```python
from main import BotEntrenadorGeneral

# Configuración personalizada
config = {
    'data_sources_path': './mi_proyecto/datos',
    'models_path': './mi_proyecto/modelos',
    'output_path': './mi_proyecto/resultados',
    'training_config': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.0001,
        'save_models': True,
        'early_stopping': True,
        'patience': 10
    },
    'autonomous_config': {
        'auto_discovery': True,
        'continuous_learning': True,
        'performance_threshold': 0.90,
        'max_cycles': 5
    },
    'supported_formats': ['.txt', '.csv', '.json', '.pdf', '.db', '.jpg', '.png', '.xlsx'],
    'default_models': ['cnn', 'lstm', 'rnn', 'transformer', 'autoencoder']
}

bot = BotEntrenadorGeneral(config)
```

### 2. **Configuración por Tipo de Proyecto**

#### **Proyecto de NLP**
```python
nlp_config = {
    'default_models': ['transformer', 'lstm', 'rnn'],
    'supported_formats': ['.txt', '.csv', '.json', '.pdf'],
    'training_config': {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001
    }
}
```

#### **Proyecto de Visión Computacional**
```python
cv_config = {
    'default_models': ['cnn', 'autoencoder'],
    'supported_formats': ['.jpg', '.png', '.bmp', '.tiff'],
    'training_config': {
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 0.0001
    }
}
```

#### **Proyecto Multimodal**
```python
multimodal_config = {
    'default_models': ['cnn', 'transformer', 'lstm'],
    'supported_formats': ['.txt', '.jpg', '.png', '.csv', '.json'],
    'training_config': {
        'batch_size': 24,
        'epochs': 75
    }
}
```

## 🎯 Casos de Uso Específicos

### 1. **Análisis de Sentimientos en Redes Sociales**
```python
# Configurar para análisis de sentimientos
sentiment_bot = BotEntrenadorGeneral({
    'data_sources_path': './datos_redes_sociales',
    'default_models': ['transformer', 'lstm'],
    'training_config': {
        'task_type': 'sentiment_analysis',
        'num_classes': 3,  # positivo, negativo, neutro
        'epochs': 30
    }
})

# Ejecutar pipeline específico
results = sentiment_bot.run_complete_pipeline()
```

### 2. **Clasificación de Imágenes Médicas**
```python
# Configurar para imágenes médicas
medical_bot = BotEntrenadorGeneral({
    'data_sources_path': './imagenes_medicas',
    'default_models': ['cnn'],
    'training_config': {
        'task_type': 'image_classification',
        'image_size': (224, 224),
        'num_classes': 5,
        'epochs': 100,
        'data_augmentation': True
    }
})

results = medical_bot.run_complete_pipeline()
```

### 3. **Generación de Contenido Automático**
```python
# Configurar para generación de texto
content_bot = BotEntrenadorGeneral({
    'data_sources_path': './corpus_textos',
    'default_models': ['transformer'],
    'training_config': {
        'task_type': 'text_generation',
        'max_length': 512,
        'temperature': 0.8,
        'epochs': 50
    }
})

results = content_bot.run_complete_pipeline()
```

## 🔍 Uso de Componentes Individuales

### 1. **Extractor de Datos Avanzado**
```python
from scripts.data_extraction import AdvancedDataExtractor

extractor = AdvancedDataExtractor()

# Extraer de múltiples fuentes
data_sources = ['./data/texto.txt', './data/imagen.jpg', './data/datos.csv']
for source in data_sources:
    extracted_data = extractor.extract_data(source)
    analysis = extractor.analyze_extracted_data(extracted_data)
    print(f"Análisis de {source}: {analysis}")
```

### 2. **Generador de Prompts Inteligentes**
```python
from scripts.prompt_generation import AdvancedPromptGenerator

generator = AdvancedPromptGenerator()

# Generar prompt optimizado
prompt = generator.create_optimized_prompt(
    task_type="clasificación de texto",
    context="Análisis de reviews de productos",
    examples=[
        {"input": "Este producto es excelente", "output": "positivo"},
        {"input": "No me gustó para nada", "output": "negativo"}
    ],
    optimization_level="advanced"
)

print(f"Prompt generado: {prompt}")
```

### 3. **Interacción con Modelos**
```python
from scripts.model_interaction import AdvancedModelInteraction

interaction = AdvancedModelInteraction()

# Interactuar con diferentes tareas
tasks = ['text_generation', 'text_classification', 'question_answering']
prompt = "¿Cuáles son los beneficios de la inteligencia artificial?"

for task in tasks:
    response = interaction.interact_with_model(prompt, task)
    print(f"Respuesta para {task}: {response}")
```

### 4. **Ecosistema Autónomo**
```python
from scripts.autonomous_ecosystem import AutonomousEcosystem

ecosystem = AutonomousEcosystem()

# Entrenar ecosistema completo
results = ecosystem.autonomous_training_cycle(
    data_sources=['./data/'],
    cycles=3,
    auto_optimize=True
)

print(f"Resultados del entrenamiento: {results}")
```

## 📊 Monitoreo y Análisis

### 1. **Visualización de Métricas**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Después del entrenamiento
training_history = results['training_results']['ecosystem_training']['training_history']

# Graficar pérdida
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_history['loss'])
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')

# Graficar precisión
plt.subplot(1, 2, 2)
plt.plot(training_history['accuracy'])
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')

plt.tight_layout()
plt.show()
```

### 2. **Análisis de Rendimiento**
```python
# Analizar resultados de testing
testing_results = results['testing_results']
performance_summary = testing_results['performance_summary']

for task, metrics in performance_summary.items():
    print(f"Tarea: {task}")
    print(f"  Tasa de éxito: {metrics['success_rate']:.2%}")
    print(f"  Intentos totales: {metrics['total_attempts']}")
    print()
```

## 🔄 Flujos de Trabajo Avanzados

### 1. **Pipeline de Experimentación**
```python
def run_experiment(config_name, config):
    """Ejecutar experimento con configuración específica"""
    print(f"🧪 Ejecutando experimento: {config_name}")
    
    bot = BotEntrenadorGeneral(config)
    results = bot.run_complete_pipeline()
    
    # Guardar resultados del experimento
    experiment_file = f"./experiments/{config_name}_results.json"
    with open(experiment_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Ejecutar múltiples experimentos
experiments = {
    'baseline': base_config,
    'optimized': optimized_config,
    'advanced': advanced_config
}

experiment_results = {}
for name, config in experiments.items():
    experiment_results[name] = run_experiment(name, config)
```

### 2. **Pipeline de Validación Cruzada**
```python
def cross_validation_pipeline(data_path, k_folds=5):
    """Pipeline con validación cruzada"""
    from sklearn.model_selection import KFold
    
    # Dividir datos en k folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_sources)):
        print(f"🔄 Procesando fold {fold + 1}/{k_folds}")
        
        # Configurar datos para este fold
        train_sources = [data_sources[i] for i in train_idx]
        val_sources = [data_sources[i] for i in val_idx]
        
        # Entrenar en datos de entrenamiento
        bot = BotEntrenadorGeneral()
        results = bot.train_autonomous_ecosystem(train_sources)
        
        # Validar en datos de validación
        validation_results = bot.interactive_model_testing(val_sources)
        
        fold_results.append({
            'fold': fold,
            'training_results': results,
            'validation_results': validation_results
        })
    
    return fold_results
```

## 🎛️ Personalización Avanzada

### 1. **Crear Modelos Personalizados**
```python
def create_custom_model(input_shape, num_classes):
    """Crear modelo personalizado"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Registrar modelo personalizado en el ecosistema
ecosystem = AutonomousEcosystem()
ecosystem.register_model('custom_mlp', create_custom_model)
```

### 2. **Funciones de Preprocesamiento Personalizadas**
```python
def custom_text_preprocessor(text):
    """Preprocesador de texto personalizado"""
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Limpiar texto
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Remover stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in text.split() if word not in stop_words]
    
    # Lemmatización
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Usar en el extractor
extractor = AdvancedDataExtractor()
extractor.add_custom_preprocessor('text', custom_text_preprocessor)
```

## 📈 Optimización de Rendimiento

### 1. **Configuración para GPU**
```python
import tensorflow as tf

# Configurar GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU disponible: {len(gpus)} dispositivos")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")

# Configuración optimizada para GPU
gpu_config = {
    'training_config': {
        'batch_size': 128,  # Batch size mayor para GPU
        'use_mixed_precision': True,
        'distribute_strategy': 'mirrored'  # Para múltiples GPUs
    }
}
```

### 2. **Procesamiento en Paralelo**
```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_data_extraction(data_sources, max_workers=None):
    """Extracción de datos en paralelo"""
    max_workers = max_workers or multiprocessing.cpu_count()
    
    extractor = AdvancedDataExtractor()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extractor.extract_data, source) 
                  for source in data_sources]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minutos timeout
                results.append(result)
            except Exception as e:
                print(f"Error en extracción paralela: {e}")
                results.append(None)
    
    return results
```

## 🔧 Solución de Problemas

### 1. **Problemas Comunes**

#### **Error de memoria insuficiente**
```python
# Reducir batch size
config['training_config']['batch_size'] = 16

# Usar gradient checkpointing
config['training_config']['gradient_checkpointing'] = True
```

#### **Convergencia lenta**
```python
# Ajustar learning rate
config['training_config']['learning_rate'] = 0.01

# Usar learning rate scheduler
config['training_config']['lr_scheduler'] = 'cosine_annealing'
```

#### **Overfitting**
```python
# Aumentar regularización
config['training_config']['dropout_rate'] = 0.5
config['training_config']['l2_regularization'] = 0.001

# Usar early stopping
config['training_config']['early_stopping'] = True
config['training_config']['patience'] = 15
```

### 2. **Debugging y Logging**
```python
import logging

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Activar modo debug en el bot
debug_config = config.copy()
debug_config['debug_mode'] = True
debug_config['verbose'] = True

bot = BotEntrenadorGeneral(debug_config)
```

## 🎉 Conclusión

Esta guía te proporciona las herramientas necesarias para aprovechar al máximo el **Bot Entrenador General**. Experimenta con diferentes configuraciones y adapta el sistema a tus necesidades específicas.

Para más información y ejemplos avanzados, consulta la documentación técnica en `MEJORAS_IMPLEMENTADAS.md`.

---
*¡Feliz entrenamiento de IA! 🚀*