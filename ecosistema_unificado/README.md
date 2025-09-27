# Ecosistema Unificado de Redes Neuronales

Sistema integrado que combina las funcionalidades de `project_root` y `red_neuronal` en un módulo unificado y funcional.

## Características Principales

### 🧠 Modelos Unificados
- **CNN**: Redes Neuronales Convolucionales con múltiples arquitecturas
- **RNN**: Redes Recurrentes (LSTM, GRU) para datos secuenciales
- **GAN**: Redes Generativas Adversarias con variantes avanzadas
- **VAE**: Autoencoders Variacionales para generación de datos
- **Transformer**: Modelos de atención para procesamiento de secuencias
- **Attention**: Mecanismos de atención especializados
- **Graph**: Redes Neuronales para datos de grafos

### 📊 Procesamiento de Datos
- Colector de datos unificado (MNIST, CIFAR-10, datos sintéticos)
- Procesador avanzado con normalización, augmentación y balanceo
- Soporte para múltiples formatos de datos

### 🔬 Evolución y Optimización
- Sistema de evolución automática de arquitecturas
- Auto-optimización de hiperparámetros
- Múltiples algoritmos de búsqueda (random, grid, bayesiano)

## Instalación

```bash
# Clonar o descargar el proyecto
cd ecosistema_unificado

# Instalar dependencias
pip install -r requirements.txt
```

## Uso Rápido

```python
# Ejecutar demostración completa
python main.py
```

## Uso Avanzado

```python
from main import EcosistemaUnificado

# Crear instancia del ecosistema
ecosistema = EcosistemaUnificado()

# Cargar datos
ecosistema.load_dataset('mnist')

# Crear y entrenar modelo
model = ecosistema.create_model('cnn', architecture='advanced')
ecosistema.train_model(model, epochs=10)

# Evaluar modelo
results = ecosistema.evaluate_model(model)

# Optimización automática
best_params = ecosistema.run_optimization('cnn', method='random_search')

# Evolución de arquitecturas
best_arch = ecosistema.run_evolution(generations=10)
```

## Estructura del Proyecto

```
ecosistema_unificado/
├── main.py                 # Sistema principal
├── requirements.txt        # Dependencias
├── README.md              # Documentación
├── models/                # Modelos unificados
│   ├── __init__.py
│   ├── cnn_unified.py
│   ├── rnn_unified.py
│   ├── gan_unified.py
│   ├── transformer_unified.py
│   ├── vae_unified.py
│   ├── attention_unified.py
│   └── graph_unified.py
├── data_processing/       # Procesamiento de datos
│   ├── __init__.py
│   ├── data_collector_unified.py
│   └── data_processor_unified.py
├── evolution/            # Sistema de evolución
│   ├── __init__.py
│   └── evolution_unified.py
└── optimization/         # Auto-optimización
    ├── __init__.py
    └── auto_optimization_unified.py
```

## Funcionalidades Destacadas

### 1. Integración Completa
- Todos los componentes trabajan de manera cohesiva
- Interfaz unificada para todas las funcionalidades
- Logging integrado para seguimiento de procesos

### 2. Flexibilidad
- Soporte para múltiples tipos de datos
- Arquitecturas configurables
- Parámetros ajustables en tiempo de ejecución

### 3. Automatización
- Evolución automática de arquitecturas
- Optimización de hiperparámetros sin intervención manual
- Procesamiento automático de datos

### 4. Escalabilidad
- Diseño modular para fácil extensión
- Soporte para nuevos modelos y algoritmos
- Configuración flexible de recursos

## Ejemplos de Uso

### Crear un CNN Avanzado
```python
cnn_model = ecosistema.create_model('cnn', 
                                   architecture='advanced',
                                   num_filters=[64, 128, 256],
                                   dropout_rate=0.3)
```

### Entrenar un GAN
```python
gan_model = ecosistema.create_model('gan', 
                                   gan_type='dcgan',
                                   latent_dim=100)
ecosistema.train_model(gan_model, epochs=50)
```

### Optimización Automática
```python
best_params = ecosistema.run_optimization(
    model_type='rnn',
    method='bayesian',
    n_trials=20
)
```

## Logs y Monitoreo

El sistema genera logs detallados en `ecosistema_unificado.log` para:
- Seguimiento de entrenamientos
- Resultados de optimización
- Errores y debugging
- Métricas de rendimiento

## Contribución

Este ecosistema está diseñado para ser extensible. Para agregar nuevos modelos:

1. Crear el archivo del modelo en `models/`
2. Implementar la interfaz estándar (train, evaluate, predict)
3. Agregar la importación en `models/__init__.py`
4. Registrar en `main.py`

## Versión

**Versión 1.0** - Ecosistema Unificado Completo

---

*Desarrollado como sistema integrado de redes neuronales avanzadas*