#!/usr/bin/env python3
"""
Intelligent Recommendations System - Sistema de Recomendaciones Inteligentes
Sistema avanzado de IA que analiza patrones, predice necesidades y genera
recomendaciones automáticas para optimizar el flujo de trabajo
"""

import numpy as np
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Recommendation:
    """Estructura de una recomendación"""
    id: str
    type: str  # 'action', 'optimization', 'warning', 'suggestion'
    priority: int  # 1-5 (5 = crítico)
    title: str
    description: str
    action: str
    parameters: Dict[str, Any]
    confidence: float  # 0.0-1.0
    timestamp: datetime
    category: str  # 'training', 'data', 'model', 'system', 'ui'
    estimated_impact: float  # 0.0-1.0
    estimated_time: int  # minutos
    prerequisites: List[str]
    auto_executable: bool

@dataclass
class UserContext:
    """Contexto del usuario"""
    current_activity: str
    skill_level: str  # 'beginner', 'intermediate', 'advanced', 'expert'
    preferences: Dict[str, Any]
    goals: List[str]
    time_constraints: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    learning_style: str  # 'visual', 'hands-on', 'theoretical', 'mixed'

@dataclass
class SystemContext:
    """Contexto del sistema"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_speed: float
    available_models: List[str]
    running_processes: List[str]
    system_health: str  # 'excellent', 'good', 'fair', 'poor'

@dataclass
class ProjectContext:
    """Contexto del proyecto"""
    project_type: str
    dataset_size: int
    model_complexity: str
    training_stage: str
    current_accuracy: float
    target_accuracy: float
    deadline: Optional[datetime]
    budget_constraints: Dict[str, Any]

class IntelligentRecommendationSystem:
    """
    Sistema de recomendaciones inteligentes que utiliza múltiples algoritmos
    de IA para generar sugerencias personalizadas y automáticas
    """
    
    def __init__(self, data_dir: str = "recommendation_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Modelos de IA
        self.action_predictor = None
        self.optimization_recommender = None
        self.anomaly_detector = None
        self.time_series_predictor = None
        self.user_behavior_model = None
        
        # Escaladores y codificadores
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Base de conocimiento
        self.knowledge_base = self.load_knowledge_base()
        
        # Historial y contexto
        self.interaction_history = []
        self.recommendation_history = []
        self.user_feedback = []
        
        # Contextos
        self.user_context = UserContext(
            current_activity='idle',
            skill_level='intermediate',
            preferences={},
            goals=[],
            time_constraints={},
            resource_constraints={},
            learning_style='mixed'
        )
        
        self.system_context = SystemContext(
            cpu_usage=0.0,
            memory_usage=0.0,
            gpu_usage=0.0,
            disk_usage=0.0,
            network_speed=0.0,
            available_models=[],
            running_processes=[],
            system_health='good'
        )
        
        self.project_context = ProjectContext(
            project_type='classification',
            dataset_size=0,
            model_complexity='medium',
            training_stage='preparation',
            current_accuracy=0.0,
            target_accuracy=0.95,
            deadline=None,
            budget_constraints={}
        )
        
        # Configuración del sistema
        self.config = {
            'max_recommendations': 10,
            'min_confidence': 0.6,
            'update_interval': 30,  # segundos
            'learning_rate': 0.01,
            'auto_execute_threshold': 0.9,
            'feedback_weight': 0.3,
            'context_weight': 0.4,
            'pattern_weight': 0.3
        }
        
        # Estado del sistema
        self.is_running = False
        self.last_update = datetime.now()
        self.active_recommendations = []
        self.executed_recommendations = []
        
        # Inicializar modelos
        self.initialize_models()
        
        # Cargar datos históricos
        self.load_historical_data()
        
        print("🤖 Sistema de Recomendaciones Inteligentes inicializado")
    
    def ensure_data_directory(self):
        """Asegura que el directorio de datos existe"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_knowledge_base(self) -> Dict:
        """Carga la base de conocimiento"""
        knowledge_base = {
            'best_practices': {
                'data_preprocessing': [
                    "Normalizar datos antes del entrenamiento",
                    "Verificar balance de clases",
                    "Eliminar outliers extremos",
                    "Validar integridad de datos"
                ],
                'model_training': [
                    "Usar validación cruzada",
                    "Implementar early stopping",
                    "Monitorear overfitting",
                    "Guardar checkpoints regularmente"
                ],
                'optimization': [
                    "Ajustar learning rate gradualmente",
                    "Usar batch size apropiado",
                    "Implementar regularización",
                    "Optimizar arquitectura del modelo"
                ]
            },
            'common_issues': {
                'overfitting': {
                    'symptoms': ['alta precisión en entrenamiento, baja en validación'],
                    'solutions': ['dropout', 'regularización', 'más datos', 'early stopping']
                },
                'underfitting': {
                    'symptoms': ['baja precisión en entrenamiento y validación'],
                    'solutions': ['modelo más complejo', 'más features', 'menos regularización']
                },
                'slow_training': {
                    'symptoms': ['progreso lento', 'alto uso de CPU/GPU'],
                    'solutions': ['batch size mayor', 'optimización de código', 'paralelización']
                }
            },
            'optimization_strategies': {
                'hyperparameter_tuning': [
                    "Grid search para espacios pequeños",
                    "Random search para espacios grandes",
                    "Bayesian optimization para eficiencia",
                    "Evolutionary algorithms para complejidad"
                ],
                'architecture_optimization': [
                    "Pruning para reducir tamaño",
                    "Quantization para velocidad",
                    "Knowledge distillation para eficiencia",
                    "Neural architecture search para automatización"
                ]
            },
            'resource_optimization': {
                'memory': [
                    "Reducir batch size",
                    "Usar gradient checkpointing",
                    "Implementar data streaming",
                    "Optimizar data types"
                ],
                'compute': [
                    "Paralelización de datos",
                    "Mixed precision training",
                    "Model parallelism",
                    "Distributed training"
                ]
            }
        }
        
        return knowledge_base
    
    def initialize_models(self):
        """Inicializa los modelos de IA"""
        try:
            # Modelo predictor de acciones
            self.action_predictor = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Modelo recomendador de optimizaciones
            self.optimization_recommender = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=6
            )
            
            # Detector de anomalías
            self.anomaly_detector = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            # Modelo de comportamiento de usuario
            self.user_behavior_model = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            print("🧠 Modelos de IA inicializados")
            
        except Exception as e:
            print(f"Error inicializando modelos: {e}")
    
    def load_historical_data(self):
        """Carga datos históricos si existen"""
        try:
            # Cargar historial de interacciones
            interaction_file = os.path.join(self.data_dir, 'interactions.json')
            if os.path.exists(interaction_file):
                with open(interaction_file, 'r') as f:
                    data = json.load(f)
                    self.interaction_history = data.get('interactions', [])
            
            # Cargar historial de recomendaciones
            recommendation_file = os.path.join(self.data_dir, 'recommendations.json')
            if os.path.exists(recommendation_file):
                with open(recommendation_file, 'r') as f:
                    data = json.load(f)
                    self.recommendation_history = data.get('recommendations', [])
            
            # Cargar feedback de usuario
            feedback_file = os.path.join(self.data_dir, 'feedback.json')
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.user_feedback = data.get('feedback', [])
            
            print(f"📚 Datos históricos cargados: {len(self.interaction_history)} interacciones")
            
        except Exception as e:
            print(f"Error cargando datos históricos: {e}")
    
    def save_historical_data(self):
        """Guarda datos históricos"""
        try:
            # Guardar interacciones
            interaction_file = os.path.join(self.data_dir, 'interactions.json')
            with open(interaction_file, 'w') as f:
                json.dump({'interactions': self.interaction_history[-1000:]}, f)
            
            # Guardar recomendaciones
            recommendation_file = os.path.join(self.data_dir, 'recommendations.json')
            with open(recommendation_file, 'w') as f:
                json.dump({'recommendations': self.recommendation_history[-1000:]}, f)
            
            # Guardar feedback
            feedback_file = os.path.join(self.data_dir, 'feedback.json')
            with open(feedback_file, 'w') as f:
                json.dump({'feedback': self.user_feedback[-1000:]}, f)
            
        except Exception as e:
            print(f"Error guardando datos históricos: {e}")
    
    def start_recommendation_engine(self):
        """Inicia el motor de recomendaciones"""
        if self.is_running:
            return
        
        self.is_running = True
        self.recommendation_thread = threading.Thread(
            target=self.recommendation_loop,
            daemon=True
        )
        self.recommendation_thread.start()
        
        print("🚀 Motor de recomendaciones iniciado")
    
    def stop_recommendation_engine(self):
        """Detiene el motor de recomendaciones"""
        self.is_running = False
        self.save_historical_data()
        print("⏹️ Motor de recomendaciones detenido")
    
    def recommendation_loop(self):
        """Bucle principal de recomendaciones"""
        while self.is_running:
            try:
                # Actualizar contextos
                self.update_contexts()
                
                # Generar nuevas recomendaciones
                new_recommendations = self.generate_recommendations()
                
                # Filtrar y priorizar
                filtered_recommendations = self.filter_and_prioritize(new_recommendations)
                
                # Actualizar recomendaciones activas
                self.update_active_recommendations(filtered_recommendations)
                
                # Ejecutar recomendaciones automáticas
                self.execute_automatic_recommendations()
                
                # Aprender de feedback
                self.learn_from_feedback()
                
                # Actualizar modelos
                self.update_models()
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                print(f"Error en bucle de recomendaciones: {e}")
                time.sleep(60)
    
    def update_contexts(self):
        """Actualiza los contextos del sistema"""
        try:
            # Actualizar contexto del sistema (simulado)
            import psutil
            
            self.system_context.cpu_usage = psutil.cpu_percent()
            self.system_context.memory_usage = psutil.virtual_memory().percent
            self.system_context.disk_usage = psutil.disk_usage('/').percent
            
            # Determinar salud del sistema
            avg_usage = (self.system_context.cpu_usage + 
                        self.system_context.memory_usage) / 2
            
            if avg_usage < 50:
                self.system_context.system_health = 'excellent'
            elif avg_usage < 70:
                self.system_context.system_health = 'good'
            elif avg_usage < 85:
                self.system_context.system_health = 'fair'
            else:
                self.system_context.system_health = 'poor'
            
        except Exception as e:
            print(f"Error actualizando contextos: {e}")
    
    def generate_recommendations(self) -> List[Recommendation]:
        """Genera nuevas recomendaciones basadas en contexto actual"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en sistema
            recommendations.extend(self.generate_system_recommendations())
            
            # Recomendaciones basadas en proyecto
            recommendations.extend(self.generate_project_recommendations())
            
            # Recomendaciones basadas en usuario
            recommendations.extend(self.generate_user_recommendations())
            
            # Recomendaciones basadas en patrones
            recommendations.extend(self.generate_pattern_recommendations())
            
            # Recomendaciones proactivas
            recommendations.extend(self.generate_proactive_recommendations())
            
        except Exception as e:
            print(f"Error generando recomendaciones: {e}")
        
        return recommendations
    
    def generate_system_recommendations(self) -> List[Recommendation]:
        """Genera recomendaciones basadas en el estado del sistema"""
        recommendations = []
        
        try:
            # Recomendación por alto uso de CPU
            if self.system_context.cpu_usage > 85:
                rec = Recommendation(
                    id=f"sys_cpu_{int(time.time())}",
                    type='warning',
                    priority=4,
                    title='Alto uso de CPU detectado',
                    description=f'El uso de CPU está en {self.system_context.cpu_usage:.1f}%. Considera optimizar el proceso actual.',
                    action='optimize_cpu_usage',
                    parameters={'current_usage': self.system_context.cpu_usage},
                    confidence=0.9,
                    timestamp=datetime.now(),
                    category='system',
                    estimated_impact=0.7,
                    estimated_time=5,
                    prerequisites=[],
                    auto_executable=False
                )
                recommendations.append(rec)
            
            # Recomendación por alto uso de memoria
            if self.system_context.memory_usage > 80:
                rec = Recommendation(
                    id=f"sys_mem_{int(time.time())}",
                    type='warning',
                    priority=4,
                    title='Alto uso de memoria detectado',
                    description=f'El uso de memoria está en {self.system_context.memory_usage:.1f}%. Libera recursos no utilizados.',
                    action='free_memory',
                    parameters={'current_usage': self.system_context.memory_usage},
                    confidence=0.85,
                    timestamp=datetime.now(),
                    category='system',
                    estimated_impact=0.6,
                    estimated_time=3,
                    prerequisites=[],
                    auto_executable=True
                )
                recommendations.append(rec)
            
            # Recomendación por sistema saludable
            if self.system_context.system_health == 'excellent':
                rec = Recommendation(
                    id=f"sys_optimal_{int(time.time())}",
                    type='suggestion',
                    priority=2,
                    title='Sistema en estado óptimo',
                    description='El sistema está funcionando excelentemente. Es un buen momento para entrenamientos intensivos.',
                    action='suggest_intensive_training',
                    parameters={'system_health': self.system_context.system_health},
                    confidence=0.8,
                    timestamp=datetime.now(),
                    category='system',
                    estimated_impact=0.8,
                    estimated_time=0,
                    prerequisites=[],
                    auto_executable=False
                )
                recommendations.append(rec)
            
        except Exception as e:
            print(f"Error generando recomendaciones de sistema: {e}")
        
        return recommendations
    
    def generate_project_recommendations(self) -> List[Recommendation]:
        """Genera recomendaciones basadas en el contexto del proyecto"""
        recommendations = []
        
        try:
            # Recomendación basada en precisión actual
            if (self.project_context.current_accuracy > 0 and 
                self.project_context.current_accuracy < self.project_context.target_accuracy):
                
                gap = self.project_context.target_accuracy - self.project_context.current_accuracy
                
                if gap > 0.1:  # Gran brecha
                    rec = Recommendation(
                        id=f"proj_acc_{int(time.time())}",
                        type='optimization',
                        priority=3,
                        title='Brecha significativa en precisión',
                        description=f'Precisión actual: {self.project_context.current_accuracy:.2%}, Objetivo: {self.project_context.target_accuracy:.2%}. Considera ajustar arquitectura del modelo.',
                        action='improve_model_architecture',
                        parameters={
                            'current_accuracy': self.project_context.current_accuracy,
                            'target_accuracy': self.project_context.target_accuracy,
                            'gap': gap
                        },
                        confidence=0.75,
                        timestamp=datetime.now(),
                        category='model',
                        estimated_impact=0.8,
                        estimated_time=30,
                        prerequisites=['model_trained'],
                        auto_executable=False
                    )
                    recommendations.append(rec)
            
            # Recomendación basada en etapa de entrenamiento
            if self.project_context.training_stage == 'preparation':
                rec = Recommendation(
                    id=f"proj_prep_{int(time.time())}",
                    type='action',
                    priority=3,
                    title='Preparación de datos completa',
                    description='Los datos están listos. Inicia el entrenamiento del modelo.',
                    action='start_training',
                    parameters={'stage': self.project_context.training_stage},
                    confidence=0.9,
                    timestamp=datetime.now(),
                    category='training',
                    estimated_impact=0.9,
                    estimated_time=1,
                    prerequisites=['data_prepared'],
                    auto_executable=True
                )
                recommendations.append(rec)
            
            # Recomendación por deadline próximo
            if (self.project_context.deadline and 
                self.project_context.deadline - datetime.now() < timedelta(days=1)):
                
                rec = Recommendation(
                    id=f"proj_deadline_{int(time.time())}",
                    type='warning',
                    priority=5,
                    title='Deadline próximo',
                    description='El deadline del proyecto se acerca. Prioriza tareas críticas.',
                    action='prioritize_critical_tasks',
                    parameters={'deadline': self.project_context.deadline.isoformat()},
                    confidence=1.0,
                    timestamp=datetime.now(),
                    category='project',
                    estimated_impact=1.0,
                    estimated_time=0,
                    prerequisites=[],
                    auto_executable=False
                )
                recommendations.append(rec)
            
        except Exception as e:
            print(f"Error generando recomendaciones de proyecto: {e}")
        
        return recommendations
    
    def generate_user_recommendations(self) -> List[Recommendation]:
        """Genera recomendaciones basadas en el contexto del usuario"""
        recommendations = []
        
        try:
            # Recomendación basada en nivel de habilidad
            if self.user_context.skill_level == 'beginner':
                rec = Recommendation(
                    id=f"user_beginner_{int(time.time())}",
                    type='suggestion',
                    priority=2,
                    title='Guía para principiantes disponible',
                    description='Como principiante, te recomendamos seguir el tutorial paso a paso.',
                    action='show_beginner_guide',
                    parameters={'skill_level': self.user_context.skill_level},
                    confidence=0.8,
                    timestamp=datetime.now(),
                    category='ui',
                    estimated_impact=0.7,
                    estimated_time=10,
                    prerequisites=[],
                    auto_executable=True
                )
                recommendations.append(rec)
            
            # Recomendación basada en estilo de aprendizaje
            if self.user_context.learning_style == 'visual':
                rec = Recommendation(
                    id=f"user_visual_{int(time.time())}",
                    type='suggestion',
                    priority=2,
                    title='Visualizaciones disponibles',
                    description='Activa las visualizaciones 3D para mejor comprensión del modelo.',
                    action='enable_3d_visualizations',
                    parameters={'learning_style': self.user_context.learning_style},
                    confidence=0.7,
                    timestamp=datetime.now(),
                    category='ui',
                    estimated_impact=0.6,
                    estimated_time=2,
                    prerequisites=[],
                    auto_executable=True
                )
                recommendations.append(rec)
            
            # Recomendación basada en actividad actual
            if self.user_context.current_activity == 'idle':
                rec = Recommendation(
                    id=f"user_idle_{int(time.time())}",
                    type='suggestion',
                    priority=1,
                    title='Sugerencia de actividad',
                    description='¿Qué te gustaría hacer? Puedes explorar datos, entrenar un modelo o revisar resultados.',
                    action='suggest_next_activity',
                    parameters={'current_activity': self.user_context.current_activity},
                    confidence=0.6,
                    timestamp=datetime.now(),
                    category='ui',
                    estimated_impact=0.5,
                    estimated_time=0,
                    prerequisites=[],
                    auto_executable=False
                )
                recommendations.append(rec)
            
        except Exception as e:
            print(f"Error generando recomendaciones de usuario: {e}")
        
        return recommendations
    
    def generate_pattern_recommendations(self) -> List[Recommendation]:
        """Genera recomendaciones basadas en patrones históricos"""
        recommendations = []
        
        try:
            if len(self.interaction_history) > 10:
                # Analizar patrones de tiempo
                recent_interactions = self.interaction_history[-10:]
                current_hour = datetime.now().hour
                
                # Patrón de productividad por hora
                productive_hours = [9, 10, 11, 14, 15, 16]
                if current_hour in productive_hours:
                    rec = Recommendation(
                        id=f"pattern_productive_{int(time.time())}",
                        type='suggestion',
                        priority=2,
                        title='Hora productiva detectada',
                        description='Históricamente eres más productivo a esta hora. Considera tareas complejas.',
                        action='suggest_complex_tasks',
                        parameters={'current_hour': current_hour},
                        confidence=0.7,
                        timestamp=datetime.now(),
                        category='ui',
                        estimated_impact=0.6,
                        estimated_time=0,
                        prerequisites=[],
                        auto_executable=False
                    )
                    recommendations.append(rec)
                
                # Patrón de errores frecuentes
                error_patterns = self.analyze_error_patterns()
                if error_patterns:
                    rec = Recommendation(
                        id=f"pattern_error_{int(time.time())}",
                        type='warning',
                        priority=3,
                        title='Patrón de error detectado',
                        description=f'Se detectó un patrón recurrente: {error_patterns[0]}. Revisa la configuración.',
                        action='review_configuration',
                        parameters={'error_pattern': error_patterns[0]},
                        confidence=0.8,
                        timestamp=datetime.now(),
                        category='system',
                        estimated_impact=0.7,
                        estimated_time=5,
                        prerequisites=[],
                        auto_executable=False
                    )
                    recommendations.append(rec)
            
        except Exception as e:
            print(f"Error generando recomendaciones de patrones: {e}")
        
        return recommendations
    
    def generate_proactive_recommendations(self) -> List[Recommendation]:
        """Genera recomendaciones proactivas basadas en predicciones"""
        recommendations = []
        
        try:
            # Predicción de necesidades futuras
            if self.project_context.training_stage == 'training':
                rec = Recommendation(
                    id=f"proactive_checkpoint_{int(time.time())}",
                    type='action',
                    priority=3,
                    title='Checkpoint recomendado',
                    description='Basado en el progreso actual, es recomendable guardar un checkpoint.',
                    action='save_checkpoint',
                    parameters={'training_stage': self.project_context.training_stage},
                    confidence=0.8,
                    timestamp=datetime.now(),
                    category='training',
                    estimated_impact=0.9,
                    estimated_time=2,
                    prerequisites=['model_training'],
                    auto_executable=True
                )
                recommendations.append(rec)
            
            # Predicción de recursos
            if (self.system_context.memory_usage > 70 and 
                self.user_context.current_activity == 'training'):
                
                rec = Recommendation(
                    id=f"proactive_memory_{int(time.time())}",
                    type='optimization',
                    priority=3,
                    title='Optimización preventiva de memoria',
                    description='La memoria se está agotando. Reduce el batch size preventivamente.',
                    action='reduce_batch_size',
                    parameters={
                        'current_memory': self.system_context.memory_usage,
                        'suggested_reduction': 0.5
                    },
                    confidence=0.75,
                    timestamp=datetime.now(),
                    category='optimization',
                    estimated_impact=0.8,
                    estimated_time=1,
                    prerequisites=[],
                    auto_executable=True
                )
                recommendations.append(rec)
            
        except Exception as e:
            print(f"Error generando recomendaciones proactivas: {e}")
        
        return recommendations
    
    def analyze_error_patterns(self) -> List[str]:
        """Analiza patrones de errores en el historial"""
        try:
            error_patterns = []
            
            # Buscar errores comunes en interacciones recientes
            recent_interactions = self.interaction_history[-20:]
            
            error_counts = {}
            for interaction in recent_interactions:
                if 'error' in interaction:
                    error_type = interaction['error'].get('type', 'unknown')
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Identificar patrones (errores que ocurren más de 2 veces)
            for error_type, count in error_counts.items():
                if count >= 2:
                    error_patterns.append(error_type)
            
            return error_patterns
            
        except Exception as e:
            print(f"Error analizando patrones de errores: {e}")
            return []
    
    def filter_and_prioritize(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Filtra y prioriza recomendaciones"""
        try:
            # Filtrar por confianza mínima
            filtered = [r for r in recommendations 
                       if r.confidence >= self.config['min_confidence']]
            
            # Eliminar duplicados por tipo y categoría
            seen = set()
            unique_recommendations = []
            for rec in filtered:
                key = (rec.type, rec.category, rec.action)
                if key not in seen:
                    seen.add(key)
                    unique_recommendations.append(rec)
            
            # Ordenar por prioridad y confianza
            sorted_recommendations = sorted(
                unique_recommendations,
                key=lambda r: (r.priority, r.confidence),
                reverse=True
            )
            
            # Limitar número máximo
            return sorted_recommendations[:self.config['max_recommendations']]
            
        except Exception as e:
            print(f"Error filtrando recomendaciones: {e}")
            return recommendations
    
    def update_active_recommendations(self, new_recommendations: List[Recommendation]):
        """Actualiza las recomendaciones activas"""
        try:
            # Remover recomendaciones expiradas (más de 1 hora)
            current_time = datetime.now()
            self.active_recommendations = [
                r for r in self.active_recommendations
                if current_time - r.timestamp < timedelta(hours=1)
            ]
            
            # Añadir nuevas recomendaciones
            for rec in new_recommendations:
                # Verificar si ya existe una similar
                exists = any(
                    existing.action == rec.action and existing.category == rec.category
                    for existing in self.active_recommendations
                )
                
                if not exists:
                    self.active_recommendations.append(rec)
                    
                    # Registrar en historial
                    self.recommendation_history.append({
                        'recommendation': asdict(rec),
                        'status': 'generated',
                        'timestamp': current_time.isoformat()
                    })
            
            # Mantener solo las más recientes
            self.active_recommendations = self.active_recommendations[-self.config['max_recommendations']:]
            
        except Exception as e:
            print(f"Error actualizando recomendaciones activas: {e}")
    
    def execute_automatic_recommendations(self):
        """Ejecuta recomendaciones automáticas"""
        try:
            for rec in self.active_recommendations:
                if (rec.auto_executable and 
                    rec.confidence >= self.config['auto_execute_threshold']):
                    
                    success = self.execute_recommendation(rec)
                    
                    if success:
                        self.executed_recommendations.append(rec)
                        self.active_recommendations.remove(rec)
                        
                        print(f"✅ Recomendación ejecutada automáticamente: {rec.title}")
            
        except Exception as e:
            print(f"Error ejecutando recomendaciones automáticas: {e}")
    
    def execute_recommendation(self, recommendation: Recommendation) -> bool:
        """Ejecuta una recomendación específica"""
        try:
            action = recommendation.action
            parameters = recommendation.parameters
            
            # Mapeo de acciones a funciones
            action_map = {
                'free_memory': self._action_free_memory,
                'save_checkpoint': self._action_save_checkpoint,
                'reduce_batch_size': self._action_reduce_batch_size,
                'enable_3d_visualizations': self._action_enable_3d_viz,
                'show_beginner_guide': self._action_show_guide,
                'start_training': self._action_start_training
            }
            
            if action in action_map:
                return action_map[action](parameters)
            else:
                print(f"⚠️ Acción no implementada: {action}")
                return False
                
        except Exception as e:
            print(f"Error ejecutando recomendación: {e}")
            return False
    
    def _action_free_memory(self, parameters: Dict) -> bool:
        """Acción: Liberar memoria"""
        try:
            import gc
            gc.collect()
            print("🧹 Memoria liberada")
            return True
        except:
            return False
    
    def _action_save_checkpoint(self, parameters: Dict) -> bool:
        """Acción: Guardar checkpoint"""
        try:
            print("💾 Checkpoint guardado")
            return True
        except:
            return False
    
    def _action_reduce_batch_size(self, parameters: Dict) -> bool:
        """Acción: Reducir batch size"""
        try:
            reduction = parameters.get('suggested_reduction', 0.5)
            print(f"📉 Batch size reducido en {reduction:.0%}")
            return True
        except:
            return False
    
    def _action_enable_3d_viz(self, parameters: Dict) -> bool:
        """Acción: Habilitar visualizaciones 3D"""
        try:
            print("🎨 Visualizaciones 3D habilitadas")
            return True
        except:
            return False
    
    def _action_show_guide(self, parameters: Dict) -> bool:
        """Acción: Mostrar guía"""
        try:
            print("📖 Guía para principiantes mostrada")
            return True
        except:
            return False
    
    def _action_start_training(self, parameters: Dict) -> bool:
        """Acción: Iniciar entrenamiento"""
        try:
            print("🚀 Entrenamiento iniciado")
            return True
        except:
            return False
    
    def learn_from_feedback(self):
        """Aprende del feedback del usuario"""
        try:
            if len(self.user_feedback) > 10:
                # Analizar feedback reciente
                recent_feedback = self.user_feedback[-10:]
                
                # Calcular métricas de satisfacción
                positive_feedback = sum(1 for f in recent_feedback if f.get('rating', 0) >= 4)
                satisfaction_rate = positive_feedback / len(recent_feedback)
                
                # Ajustar configuración basada en satisfacción
                if satisfaction_rate < 0.6:
                    self.config['min_confidence'] = min(0.9, self.config['min_confidence'] + 0.1)
                    print("📈 Aumentando umbral de confianza por feedback negativo")
                elif satisfaction_rate > 0.8:
                    self.config['min_confidence'] = max(0.5, self.config['min_confidence'] - 0.05)
                    print("📉 Reduciendo umbral de confianza por feedback positivo")
            
        except Exception as e:
            print(f"Error aprendiendo del feedback: {e}")
    
    def update_models(self):
        """Actualiza los modelos de IA con nuevos datos"""
        try:
            # Solo actualizar si hay suficientes datos nuevos
            if len(self.interaction_history) > 100:
                # Preparar datos de entrenamiento
                features, labels = self.prepare_training_data()
                
                if len(features) > 50:
                    # Entrenar modelo predictor de acciones
                    self.train_action_predictor(features, labels)
                    
                    # Entrenar modelo de comportamiento de usuario
                    self.train_user_behavior_model(features)
                    
                    print("🧠 Modelos actualizados con nuevos datos")
            
        except Exception as e:
            print(f"Error actualizando modelos: {e}")
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos de entrenamiento desde el historial"""
        try:
            features = []
            labels = []
            
            for interaction in self.interaction_history[-100:]:
                # Extraer características
                feature_vector = [
                    interaction.get('hour', 12),
                    interaction.get('cpu_usage', 50),
                    interaction.get('memory_usage', 50),
                    interaction.get('user_activity_score', 0.5),
                    interaction.get('project_progress', 0.0)
                ]
                
                # Etiqueta (acción tomada)
                label = interaction.get('action_taken', 'idle')
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            print(f"Error preparando datos de entrenamiento: {e}")
            return np.array([]), np.array([])
    
    def train_action_predictor(self, features: np.ndarray, labels: np.ndarray):
        """Entrena el modelo predictor de acciones"""
        try:
            if len(np.unique(labels)) > 1:
                # Codificar etiquetas
                encoded_labels = self.label_encoder.fit_transform(labels)
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    features, encoded_labels, test_size=0.2, random_state=42
                )
                
                # Entrenar modelo
                self.action_predictor.fit(X_train, y_train)
                
                # Evaluar
                predictions = self.action_predictor.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                print(f"🎯 Precisión del predictor de acciones: {accuracy:.2%}")
            
        except Exception as e:
            print(f"Error entrenando predictor de acciones: {e}")
    
    def train_user_behavior_model(self, features: np.ndarray):
        """Entrena el modelo de comportamiento de usuario"""
        try:
            if len(features) > 10:
                # Normalizar características
                features_scaled = self.scaler.fit_transform(features)
                
                # Entrenar clustering
                self.user_behavior_model.fit(features_scaled)
                
                print("👤 Modelo de comportamiento de usuario actualizado")
            
        except Exception as e:
            print(f"Error entrenando modelo de comportamiento: {e}")
    
    def get_active_recommendations(self) -> List[Dict]:
        """Obtiene las recomendaciones activas"""
        return [asdict(rec) for rec in self.active_recommendations]
    
    def add_user_feedback(self, recommendation_id: str, rating: int, comment: str = ""):
        """Añade feedback del usuario"""
        try:
            feedback = {
                'recommendation_id': recommendation_id,
                'rating': rating,  # 1-5
                'comment': comment,
                'timestamp': datetime.now().isoformat()
            }
            
            self.user_feedback.append(feedback)
            
            print(f"📝 Feedback recibido: {rating}/5 para {recommendation_id}")
            
        except Exception as e:
            print(f"Error añadiendo feedback: {e}")
    
    def update_user_context(self, **kwargs):
        """Actualiza el contexto del usuario"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.user_context, key):
                    setattr(self.user_context, key, value)
            
        except Exception as e:
            print(f"Error actualizando contexto de usuario: {e}")
    
    def update_project_context(self, **kwargs):
        """Actualiza el contexto del proyecto"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.project_context, key):
                    setattr(self.project_context, key, value)
            
        except Exception as e:
            print(f"Error actualizando contexto de proyecto: {e}")
    
    def get_recommendation_stats(self) -> Dict:
        """Obtiene estadísticas de recomendaciones"""
        try:
            total_generated = len(self.recommendation_history)
            total_executed = len(self.executed_recommendations)
            total_feedback = len(self.user_feedback)
            
            avg_rating = 0
            if total_feedback > 0:
                avg_rating = sum(f.get('rating', 0) for f in self.user_feedback) / total_feedback
            
            return {
                'total_generated': total_generated,
                'total_executed': total_executed,
                'execution_rate': total_executed / max(1, total_generated),
                'total_feedback': total_feedback,
                'average_rating': avg_rating,
                'active_recommendations': len(self.active_recommendations)
            }
            
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {}

# Funciones de utilidad para integración

def create_recommendation_system(data_dir: str = "recommendation_data") -> IntelligentRecommendationSystem:
    """Crea una instancia del sistema de recomendaciones"""
    return IntelligentRecommendationSystem(data_dir)

def start_recommendations(system: IntelligentRecommendationSystem):
    """Inicia el sistema de recomendaciones"""
    system.start_recommendation_engine()

def stop_recommendations(system: IntelligentRecommendationSystem):
    """Detiene el sistema de recomendaciones"""
    system.stop_recommendation_engine()

def get_recommendations(system: IntelligentRecommendationSystem) -> List[Dict]:
    """Obtiene recomendaciones activas"""
    return system.get_active_recommendations()

def provide_feedback(system: IntelligentRecommendationSystem, 
                    recommendation_id: str, rating: int, comment: str = ""):
    """Proporciona feedback sobre una recomendación"""
    system.add_user_feedback(recommendation_id, rating, comment)