#!/usr/bin/env python3
"""
Adaptive Dashboard - Dashboard Adaptativo Inteligente
Sistema de dashboard que se adapta automáticamente a las necesidades del usuario
con predicción de acciones y personalización inteligente
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import psutil
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class AdaptiveDashboard:
    """
    Dashboard adaptativo inteligente que aprende de los patrones de uso
    y predice las necesidades del usuario
    """
    
    def __init__(self, parent_frame, ai_consciousness=None):
        self.parent_frame = parent_frame
        self.ai_consciousness = ai_consciousness
        
        # Configuración del dashboard
        self.dashboard_config = {
            'theme': 'dark',
            'layout': 'adaptive',
            'auto_refresh': True,
            'refresh_interval': 2.0,
            'prediction_enabled': True,
            'learning_enabled': True,
            'animation_enabled': True
        }
        
        # Estado del usuario y sistema
        self.user_state = {
            'current_activity': 'idle',
            'last_interaction': datetime.now(),
            'interaction_count': 0,
            'preferred_widgets': [],
            'usage_patterns': {},
            'attention_focus': 'center',
            'stress_level': 'low',
            'productivity_score': 0.8
        }
        
        # Métricas del sistema
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'disk_usage': 0.0,
            'network_activity': 0.0,
            'training_progress': 0.0,
            'model_accuracy': 0.0,
            'learning_rate': 0.001
        }
        
        # Widgets adaptativos
        self.adaptive_widgets = {}
        self.widget_priorities = {}
        self.widget_positions = {}
        
        # Predicciones y recomendaciones
        self.predictions = {
            'next_action': None,
            'optimal_parameters': {},
            'resource_needs': {},
            'completion_time': None,
            'success_probability': 0.0
        }
        
        # Historial de datos para aprendizaje
        self.data_history = {
            'interactions': [],
            'system_performance': [],
            'user_behavior': [],
            'model_metrics': []
        }
        
        # Configurar tema
        self.setup_theme()
        
        # Crear layout adaptativo
        self.create_adaptive_layout()
        
        # Inicializar sistemas de aprendizaje
        self.initialize_learning_systems()
        
        # Iniciar bucles de actualización
        self.start_update_loops()
        
        print("📊 Dashboard Adaptativo inicializado")
    
    def setup_theme(self):
        """Configura el tema visual del dashboard"""
        # Colores del tema futurista
        self.colors = {
            'primary': '#00D4FF',      # Azul cibernético
            'secondary': '#FF6B35',    # Naranja energético
            'accent': '#4ECDC4',       # Verde agua
            'background': '#0A0E27',   # Azul oscuro profundo
            'surface': '#1A1F3A',      # Azul gris
            'text_primary': '#FFFFFF', # Blanco
            'text_secondary': '#B0BEC5', # Gris claro
            'success': '#4CAF50',      # Verde
            'warning': '#FF9800',      # Naranja
            'error': '#F44336',        # Rojo
            'neural': '#9C27B0'        # Púrpura neural
        }
        
        # Configurar CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
    
    def create_adaptive_layout(self):
        """Crea el layout adaptativo del dashboard"""
        # Frame principal
        self.main_frame = ctk.CTkFrame(
            self.parent_frame,
            fg_color=self.colors['background']
        )
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Crear áreas del dashboard
        self.create_header_area()
        self.create_main_content_area()
        self.create_sidebar_area()
        self.create_footer_area()
        
        # Crear widgets adaptativos
        self.create_adaptive_widgets()
    
    def create_header_area(self):
        """Crea el área de encabezado"""
        self.header_frame = ctk.CTkFrame(
            self.main_frame,
            height=80,
            fg_color=self.colors['surface']
        )
        self.header_frame.pack(fill="x", padx=5, pady=(5, 2))
        self.header_frame.pack_propagate(False)
        
        # Título inteligente
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="🧠 NeuroVision AI Dashboard",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['primary']
        )
        self.title_label.pack(side="left", padx=20, pady=20)
        
        # Estado de la IA
        self.ai_status_frame = ctk.CTkFrame(self.header_frame)
        self.ai_status_frame.pack(side="right", padx=20, pady=10)
        
        self.ai_status_label = ctk.CTkLabel(
            self.ai_status_frame,
            text="🤖 IA Activa",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['success']
        )
        self.ai_status_label.pack(padx=10, pady=5)
        
        # Predicción actual
        self.prediction_label = ctk.CTkLabel(
            self.ai_status_frame,
            text="🔮 Analizando patrones...",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['text_secondary']
        )
        self.prediction_label.pack(padx=10, pady=(0, 5))
    
    def create_main_content_area(self):
        """Crea el área de contenido principal"""
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.colors['background']
        )
        self.content_frame.pack(fill="both", expand=True, padx=5, pady=2)
        
        # Crear grid adaptativo
        self.content_frame.grid_columnconfigure(0, weight=2)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)
        
        # Área de visualización principal
        self.main_viz_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.colors['surface']
        )
        self.main_viz_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))
        
        # Área de métricas
        self.metrics_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.colors['surface']
        )
        self.metrics_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=(0, 2))
        
        # Área de control
        self.control_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.colors['surface']
        )
        self.control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 2), pady=(2, 0))
        
        # Área de recomendaciones
        self.recommendations_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=self.colors['surface']
        )
        self.recommendations_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 0), pady=(2, 0))
    
    def create_sidebar_area(self):
        """Crea el área de barra lateral (oculta por defecto)"""
        self.sidebar_visible = False
        self.sidebar_frame = ctk.CTkFrame(
            self.main_frame,
            width=250,
            fg_color=self.colors['surface']
        )
        # No se empaqueta inicialmente
    
    def create_footer_area(self):
        """Crea el área de pie de página"""
        self.footer_frame = ctk.CTkFrame(
            self.main_frame,
            height=40,
            fg_color=self.colors['surface']
        )
        self.footer_frame.pack(fill="x", padx=5, pady=(2, 5))
        self.footer_frame.pack_propagate(False)
        
        # Información del sistema
        self.system_info_label = ctk.CTkLabel(
            self.footer_frame,
            text="Sistema: Listo | CPU: 0% | RAM: 0% | GPU: 0%",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['text_secondary']
        )
        self.system_info_label.pack(side="left", padx=10, pady=10)
        
        # Tiempo de sesión
        self.session_time_label = ctk.CTkLabel(
            self.footer_frame,
            text="Sesión: 00:00:00",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['text_secondary']
        )
        self.session_time_label.pack(side="right", padx=10, pady=10)
    
    def create_adaptive_widgets(self):
        """Crea widgets adaptativos"""
        # Widget de visualización principal
        self.create_main_visualization_widget()
        
        # Widget de métricas en tiempo real
        self.create_real_time_metrics_widget()
        
        # Widget de control inteligente
        self.create_intelligent_control_widget()
        
        # Widget de recomendaciones
        self.create_recommendations_widget()
    
    def create_main_visualization_widget(self):
        """Crea el widget de visualización principal"""
        # Título
        viz_title = ctk.CTkLabel(
            self.main_viz_frame,
            text="📊 Visualización Inteligente",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        viz_title.pack(pady=(10, 5))
        
        # Notebook para diferentes visualizaciones
        self.viz_notebook = ctk.CTkTabview(self.main_viz_frame)
        self.viz_notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Pestaña de entrenamiento
        self.viz_notebook.add("Entrenamiento")
        self.training_viz_frame = self.viz_notebook.tab("Entrenamiento")
        
        # Pestaña de arquitectura
        self.viz_notebook.add("Arquitectura")
        self.architecture_viz_frame = self.viz_notebook.tab("Arquitectura")
        
        # Pestaña de datos
        self.viz_notebook.add("Datos")
        self.data_viz_frame = self.viz_notebook.tab("Datos")
        
        # Crear visualizaciones
        self.create_training_visualization()
        self.create_architecture_visualization()
        self.create_data_visualization()
    
    def create_training_visualization(self):
        """Crea visualización de entrenamiento"""
        # Crear figura de matplotlib
        self.training_fig, (self.loss_ax, self.acc_ax) = plt.subplots(2, 1, figsize=(8, 6))
        self.training_fig.patch.set_facecolor(self.colors['background'])
        
        # Configurar ejes
        for ax in [self.loss_ax, self.acc_ax]:
            ax.set_facecolor(self.colors['surface'])
            ax.tick_params(colors=self.colors['text_secondary'])
            ax.spines['bottom'].set_color(self.colors['text_secondary'])
            ax.spines['top'].set_color(self.colors['text_secondary'])
            ax.spines['right'].set_color(self.colors['text_secondary'])
            ax.spines['left'].set_color(self.colors['text_secondary'])
        
        # Configurar gráficos
        self.loss_ax.set_title('Pérdida del Modelo', color=self.colors['text_primary'])
        self.loss_ax.set_ylabel('Loss', color=self.colors['text_primary'])
        
        self.acc_ax.set_title('Precisión del Modelo', color=self.colors['text_primary'])
        self.acc_ax.set_ylabel('Accuracy', color=self.colors['text_primary'])
        self.acc_ax.set_xlabel('Época', color=self.colors['text_primary'])
        
        # Integrar con tkinter
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, self.training_viz_frame)
        self.training_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Datos iniciales
        self.training_data = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def create_architecture_visualization(self):
        """Crea visualización de arquitectura"""
        # Placeholder para visualización 3D de arquitectura
        arch_label = ctk.CTkLabel(
            self.architecture_viz_frame,
            text="🏗️ Visualización 3D de Arquitectura\n(Integración con holographic_visualizer.py)",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['text_secondary']
        )
        arch_label.pack(expand=True)
    
    def create_data_visualization(self):
        """Crea visualización de datos"""
        # Placeholder para exploración de datos
        data_label = ctk.CTkLabel(
            self.data_viz_frame,
            text="📈 Exploración Inteligente de Datos\n(t-SNE, PCA, Distribuciones)",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['text_secondary']
        )
        data_label.pack(expand=True)
    
    def create_real_time_metrics_widget(self):
        """Crea widget de métricas en tiempo real"""
        # Título
        metrics_title = ctk.CTkLabel(
            self.metrics_frame,
            text="⚡ Métricas en Tiempo Real",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        metrics_title.pack(pady=(10, 5))
        
        # Scrollable frame para métricas
        self.metrics_scroll = ctk.CTkScrollableFrame(self.metrics_frame)
        self.metrics_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Métricas del sistema
        self.create_metric_display("CPU", "cpu_usage", "%")
        self.create_metric_display("RAM", "memory_usage", "%")
        self.create_metric_display("GPU", "gpu_usage", "%")
        
        # Métricas del modelo
        self.create_metric_display("Precisión", "model_accuracy", "%")
        self.create_metric_display("Progreso", "training_progress", "%")
        self.create_metric_display("Learning Rate", "learning_rate", "")
        
        # Métricas de usuario
        self.create_metric_display("Productividad", "productivity_score", "")
        self.create_metric_display("Interacciones", "interaction_count", "")
    
    def create_metric_display(self, name: str, key: str, unit: str):
        """Crea un display de métrica individual"""
        metric_frame = ctk.CTkFrame(self.metrics_scroll)
        metric_frame.pack(fill="x", pady=2)
        
        # Nombre de la métrica
        name_label = ctk.CTkLabel(
            metric_frame,
            text=name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors['text_primary']
        )
        name_label.pack(side="left", padx=(10, 5), pady=5)
        
        # Valor de la métrica
        value_label = ctk.CTkLabel(
            metric_frame,
            text=f"0{unit}",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['accent']
        )
        value_label.pack(side="right", padx=(5, 10), pady=5)
        
        # Barra de progreso
        if unit == "%":
            progress_bar = ctk.CTkProgressBar(
                metric_frame,
                width=100,
                height=10
            )
            progress_bar.pack(side="right", padx=(5, 10), pady=5)
            progress_bar.set(0)
            
            # Guardar referencias
            self.adaptive_widgets[f"{key}_progress"] = progress_bar
        
        self.adaptive_widgets[f"{key}_label"] = value_label
    
    def create_intelligent_control_widget(self):
        """Crea widget de control inteligente"""
        # Título
        control_title = ctk.CTkLabel(
            self.control_frame,
            text="🎮 Control Inteligente",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        control_title.pack(pady=(10, 5))
        
        # Frame de botones
        buttons_frame = ctk.CTkFrame(self.control_frame)
        buttons_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Botones inteligentes
        self.smart_train_btn = ctk.CTkButton(
            buttons_frame,
            text="🧠 Entrenamiento Inteligente",
            command=self.start_intelligent_training,
            fg_color=self.colors['primary'],
            hover_color=self.colors['accent']
        )
        self.smart_train_btn.pack(fill="x", padx=5, pady=5)
        
        self.auto_optimize_btn = ctk.CTkButton(
            buttons_frame,
            text="⚡ Auto-Optimización",
            command=self.start_auto_optimization,
            fg_color=self.colors['secondary'],
            hover_color=self.colors['warning']
        )
        self.auto_optimize_btn.pack(fill="x", padx=5, pady=5)
        
        self.predict_btn = ctk.CTkButton(
            buttons_frame,
            text="🔮 Predicción Inteligente",
            command=self.generate_predictions,
            fg_color=self.colors['neural'],
            hover_color=self.colors['accent']
        )
        self.predict_btn.pack(fill="x", padx=5, pady=5)
        
        # Configuración adaptativa
        config_frame = ctk.CTkFrame(self.control_frame)
        config_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Switch de aprendizaje automático
        self.auto_learning_switch = ctk.CTkSwitch(
            config_frame,
            text="Aprendizaje Automático",
            command=self.toggle_auto_learning
        )
        self.auto_learning_switch.pack(padx=10, pady=5)
        self.auto_learning_switch.select()
        
        # Switch de predicciones
        self.predictions_switch = ctk.CTkSwitch(
            config_frame,
            text="Predicciones Activas",
            command=self.toggle_predictions
        )
        self.predictions_switch.pack(padx=10, pady=5)
        self.predictions_switch.select()
    
    def create_recommendations_widget(self):
        """Crea widget de recomendaciones"""
        # Título
        rec_title = ctk.CTkLabel(
            self.recommendations_frame,
            text="💡 Recomendaciones IA",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        rec_title.pack(pady=(10, 5))
        
        # Área de recomendaciones
        self.recommendations_text = ctk.CTkTextbox(
            self.recommendations_frame,
            height=200,
            font=ctk.CTkFont(size=11)
        )
        self.recommendations_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Recomendaciones iniciales
        initial_recommendations = """🤖 IA Iniciada - Analizando patrones...

💡 Recomendaciones Inteligentes:
• Sistema listo para entrenamiento
• Datos disponibles para análisis
• Configuración óptima detectada

🔮 Predicciones:
• Tiempo estimado de entrenamiento: Calculando...
• Precisión esperada: Analizando...
• Recursos necesarios: Evaluando...

⚡ Acciones Sugeridas:
• Iniciar entrenamiento inteligente
• Activar auto-optimización
• Configurar callbacks automáticos
"""
        
        self.recommendations_text.insert("1.0", initial_recommendations)
        self.recommendations_text.configure(state="disabled")
    
    def initialize_learning_systems(self):
        """Inicializa los sistemas de aprendizaje"""
        # Modelo de predicción de comportamiento del usuario
        self.user_behavior_model = None
        
        # Scaler para normalización de datos
        self.scaler = StandardScaler()
        
        # Clustering para patrones de uso
        self.usage_clusterer = KMeans(n_clusters=5, random_state=42)
        
        # Historial de sesiones
        self.session_start_time = datetime.now()
        
        print("🧠 Sistemas de aprendizaje inicializados")
    
    def start_update_loops(self):
        """Inicia los bucles de actualización"""
        # Bucle de actualización de métricas
        self.metrics_thread = threading.Thread(target=self.metrics_update_loop, daemon=True)
        self.metrics_thread.start()
        
        # Bucle de aprendizaje y predicción
        self.learning_thread = threading.Thread(target=self.learning_update_loop, daemon=True)
        self.learning_thread.start()
        
        # Bucle de adaptación de UI
        self.adaptation_thread = threading.Thread(target=self.ui_adaptation_loop, daemon=True)
        self.adaptation_thread.start()
    
    def metrics_update_loop(self):
        """Bucle de actualización de métricas del sistema"""
        while True:
            try:
                # Actualizar métricas del sistema
                self.update_system_metrics()
                
                # Actualizar displays
                self.update_metric_displays()
                
                # Actualizar visualizaciones
                self.update_visualizations()
                
                time.sleep(self.dashboard_config['refresh_interval'])
                
            except Exception as e:
                print(f"Error en actualización de métricas: {e}")
                time.sleep(5)
    
    def learning_update_loop(self):
        """Bucle de aprendizaje y predicción"""
        while True:
            try:
                if self.dashboard_config['learning_enabled']:
                    # Aprender de patrones de uso
                    self.learn_usage_patterns()
                    
                    # Generar predicciones
                    if self.dashboard_config['prediction_enabled']:
                        self.update_predictions()
                    
                    # Actualizar recomendaciones
                    self.update_recommendations()
                
                time.sleep(10)  # Actualizar cada 10 segundos
                
            except Exception as e:
                print(f"Error en bucle de aprendizaje: {e}")
                time.sleep(15)
    
    def ui_adaptation_loop(self):
        """Bucle de adaptación de la interfaz"""
        while True:
            try:
                if self.dashboard_config['layout'] == 'adaptive':
                    # Adaptar layout basado en uso
                    self.adapt_layout()
                    
                    # Reorganizar widgets por prioridad
                    self.reorganize_widgets()
                    
                    # Ajustar tema si es necesario
                    self.adapt_theme()
                
                time.sleep(30)  # Adaptar cada 30 segundos
                
            except Exception as e:
                print(f"Error en adaptación de UI: {e}")
                time.sleep(60)
    
    def update_system_metrics(self):
        """Actualiza las métricas del sistema"""
        try:
            # Métricas del sistema
            self.system_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            self.system_metrics['memory_usage'] = psutil.virtual_memory().percent
            self.system_metrics['disk_usage'] = psutil.disk_usage('/').percent
            
            # Métricas de GPU (simuladas si no hay GPU disponible)
            try:
                # Aquí se podría integrar con nvidia-ml-py para métricas reales de GPU
                self.system_metrics['gpu_usage'] = np.random.uniform(0, 100)
            except:
                self.system_metrics['gpu_usage'] = 0
            
            # Métricas de red
            net_io = psutil.net_io_counters()
            self.system_metrics['network_activity'] = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
            
            # Actualizar información del pie de página
            self.update_footer_info()
            
        except Exception as e:
            print(f"Error actualizando métricas del sistema: {e}")
    
    def update_metric_displays(self):
        """Actualiza los displays de métricas"""
        try:
            # Actualizar métricas del sistema
            metrics_to_update = [
                ('cpu_usage', '%'),
                ('memory_usage', '%'),
                ('gpu_usage', '%'),
                ('model_accuracy', '%'),
                ('training_progress', '%')
            ]
            
            for metric_key, unit in metrics_to_update:
                if metric_key in self.system_metrics:
                    value = self.system_metrics[metric_key]
                    
                    # Actualizar label
                    if f"{metric_key}_label" in self.adaptive_widgets:
                        label = self.adaptive_widgets[f"{metric_key}_label"]
                        if unit == '%':
                            label.configure(text=f"{value:.1f}{unit}")
                        else:
                            label.configure(text=f"{value:.3f}")
                    
                    # Actualizar barra de progreso
                    if f"{metric_key}_progress" in self.adaptive_widgets:
                        progress = self.adaptive_widgets[f"{metric_key}_progress"]
                        progress.set(value / 100.0)
            
            # Actualizar métricas de usuario
            self.adaptive_widgets['interaction_count_label'].configure(
                text=str(self.user_state['interaction_count'])
            )
            
            self.adaptive_widgets['productivity_score_label'].configure(
                text=f"{self.user_state['productivity_score']:.2f}"
            )
            
        except Exception as e:
            print(f"Error actualizando displays de métricas: {e}")
    
    def update_visualizations(self):
        """Actualiza las visualizaciones"""
        try:
            # Simular datos de entrenamiento
            if len(self.training_data['epochs']) < 100:  # Simular hasta 100 épocas
                epoch = len(self.training_data['epochs']) + 1
                
                # Simular pérdida decreciente con ruido
                loss = 2.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.1)
                val_loss = loss + np.random.normal(0, 0.05)
                
                # Simular precisión creciente
                accuracy = 1 - np.exp(-epoch / 15) + np.random.normal(0, 0.02)
                val_accuracy = accuracy + np.random.normal(0, 0.01)
                
                # Añadir datos
                self.training_data['epochs'].append(epoch)
                self.training_data['loss'].append(max(0, loss))
                self.training_data['val_loss'].append(max(0, val_loss))
                self.training_data['accuracy'].append(min(1, max(0, accuracy)))
                self.training_data['val_accuracy'].append(min(1, max(0, val_accuracy)))
                
                # Actualizar gráficos
                self.update_training_plots()
                
                # Actualizar métricas del modelo
                self.system_metrics['model_accuracy'] = accuracy * 100
                self.system_metrics['training_progress'] = (epoch / 100) * 100
            
        except Exception as e:
            print(f"Error actualizando visualizaciones: {e}")
    
    def update_training_plots(self):
        """Actualiza los gráficos de entrenamiento"""
        try:
            # Limpiar ejes
            self.loss_ax.clear()
            self.acc_ax.clear()
            
            # Configurar colores de fondo
            self.loss_ax.set_facecolor(self.colors['surface'])
            self.acc_ax.set_facecolor(self.colors['surface'])
            
            # Graficar pérdida
            self.loss_ax.plot(
                self.training_data['epochs'],
                self.training_data['loss'],
                color=self.colors['error'],
                label='Training Loss',
                linewidth=2
            )
            self.loss_ax.plot(
                self.training_data['epochs'],
                self.training_data['val_loss'],
                color=self.colors['warning'],
                label='Validation Loss',
                linewidth=2,
                linestyle='--'
            )
            
            # Graficar precisión
            self.acc_ax.plot(
                self.training_data['epochs'],
                self.training_data['accuracy'],
                color=self.colors['success'],
                label='Training Accuracy',
                linewidth=2
            )
            self.acc_ax.plot(
                self.training_data['epochs'],
                self.training_data['val_accuracy'],
                color=self.colors['accent'],
                label='Validation Accuracy',
                linewidth=2,
                linestyle='--'
            )
            
            # Configurar ejes
            for ax in [self.loss_ax, self.acc_ax]:
                ax.tick_params(colors=self.colors['text_secondary'])
                ax.spines['bottom'].set_color(self.colors['text_secondary'])
                ax.spines['top'].set_color(self.colors['text_secondary'])
                ax.spines['right'].set_color(self.colors['text_secondary'])
                ax.spines['left'].set_color(self.colors['text_secondary'])
                ax.legend(facecolor=self.colors['surface'], 
                         edgecolor=self.colors['text_secondary'],
                         labelcolor=self.colors['text_secondary'])
            
            # Títulos y etiquetas
            self.loss_ax.set_title('Pérdida del Modelo', color=self.colors['text_primary'])
            self.loss_ax.set_ylabel('Loss', color=self.colors['text_primary'])
            
            self.acc_ax.set_title('Precisión del Modelo', color=self.colors['text_primary'])
            self.acc_ax.set_ylabel('Accuracy', color=self.colors['text_primary'])
            self.acc_ax.set_xlabel('Época', color=self.colors['text_primary'])
            
            # Actualizar canvas
            self.training_canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráficos de entrenamiento: {e}")
    
    def update_footer_info(self):
        """Actualiza la información del pie de página"""
        try:
            # Información del sistema
            cpu = self.system_metrics['cpu_usage']
            ram = self.system_metrics['memory_usage']
            gpu = self.system_metrics['gpu_usage']
            
            system_text = f"Sistema: Activo | CPU: {cpu:.1f}% | RAM: {ram:.1f}% | GPU: {gpu:.1f}%"
            self.system_info_label.configure(text=system_text)
            
            # Tiempo de sesión
            session_duration = datetime.now() - self.session_start_time
            hours, remainder = divmod(session_duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            session_text = f"Sesión: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            self.session_time_label.configure(text=session_text)
            
        except Exception as e:
            print(f"Error actualizando información del pie: {e}")
    
    def learn_usage_patterns(self):
        """Aprende patrones de uso del usuario"""
        try:
            # Registrar interacción actual
            current_time = datetime.now()
            interaction_data = {
                'timestamp': current_time,
                'activity': self.user_state['current_activity'],
                'cpu_usage': self.system_metrics['cpu_usage'],
                'memory_usage': self.system_metrics['memory_usage'],
                'interaction_count': self.user_state['interaction_count']
            }
            
            self.data_history['user_behavior'].append(interaction_data)
            
            # Mantener solo los últimos 1000 registros
            if len(self.data_history['user_behavior']) > 1000:
                self.data_history['user_behavior'] = self.data_history['user_behavior'][-1000:]
            
            # Analizar patrones si hay suficientes datos
            if len(self.data_history['user_behavior']) > 50:
                self.analyze_usage_patterns()
            
        except Exception as e:
            print(f"Error aprendiendo patrones de uso: {e}")
    
    def analyze_usage_patterns(self):
        """Analiza patrones de uso para predicciones"""
        try:
            # Preparar datos para análisis
            behavior_data = self.data_history['user_behavior'][-100:]  # Últimos 100 registros
            
            # Extraer características
            features = []
            for record in behavior_data:
                hour = record['timestamp'].hour
                minute = record['timestamp'].minute
                cpu = record['cpu_usage']
                memory = record['memory_usage']
                interactions = record['interaction_count']
                
                features.append([hour, minute, cpu, memory, interactions])
            
            features = np.array(features)
            
            # Normalizar datos
            if len(features) > 10:
                features_scaled = self.scaler.fit_transform(features)
                
                # Clustering para identificar patrones
                clusters = self.usage_clusterer.fit_predict(features_scaled)
                
                # Actualizar patrones de uso
                current_cluster = clusters[-1]
                self.user_state['usage_patterns']['current_cluster'] = current_cluster
                
                # Calcular productividad basada en patrones
                self.calculate_productivity_score(features, clusters)
            
        except Exception as e:
            print(f"Error analizando patrones de uso: {e}")
    
    def calculate_productivity_score(self, features, clusters):
        """Calcula puntuación de productividad"""
        try:
            # Calcular productividad basada en interacciones y recursos
            recent_interactions = features[-10:, 4]  # Últimas 10 interacciones
            recent_cpu = features[-10:, 2]  # Último uso de CPU
            
            # Productividad = interacciones / uso de recursos
            if np.mean(recent_cpu) > 0:
                productivity = np.mean(recent_interactions) / (np.mean(recent_cpu) / 100)
                productivity = min(1.0, productivity / 10)  # Normalizar a 0-1
            else:
                productivity = 0.5
            
            self.user_state['productivity_score'] = productivity
            
        except Exception as e:
            print(f"Error calculando productividad: {e}")
    
    def update_predictions(self):
        """Actualiza predicciones inteligentes"""
        try:
            # Predecir próxima acción basada en patrones
            self.predict_next_action()
            
            # Predecir recursos necesarios
            self.predict_resource_needs()
            
            # Predecir tiempo de finalización
            self.predict_completion_time()
            
            # Actualizar estado de predicciones en header
            self.update_prediction_display()
            
        except Exception as e:
            print(f"Error actualizando predicciones: {e}")
    
    def predict_next_action(self):
        """Predice la próxima acción del usuario"""
        try:
            current_hour = datetime.now().hour
            current_activity = self.user_state['current_activity']
            
            # Predicciones simples basadas en patrones comunes
            if current_hour < 12:
                self.predictions['next_action'] = 'data_exploration'
            elif current_hour < 18:
                self.predictions['next_action'] = 'model_training'
            else:
                self.predictions['next_action'] = 'results_analysis'
            
            self.predictions['success_probability'] = 0.75
            
        except Exception as e:
            print(f"Error prediciendo próxima acción: {e}")
    
    def predict_resource_needs(self):
        """Predice necesidades de recursos"""
        try:
            current_cpu = self.system_metrics['cpu_usage']
            current_memory = self.system_metrics['memory_usage']
            
            # Predicción basada en tendencias actuales
            predicted_cpu = min(100, current_cpu * 1.2)
            predicted_memory = min(100, current_memory * 1.1)
            
            self.predictions['resource_needs'] = {
                'cpu': predicted_cpu,
                'memory': predicted_memory,
                'gpu': 80,  # Estimación para entrenamiento
                'time': '15-30 min'
            }
            
        except Exception as e:
            print(f"Error prediciendo recursos: {e}")
    
    def predict_completion_time(self):
        """Predice tiempo de finalización"""
        try:
            if self.system_metrics['training_progress'] > 0:
                progress = self.system_metrics['training_progress'] / 100
                if progress > 0:
                    elapsed_time = datetime.now() - self.session_start_time
                    total_estimated = elapsed_time / progress
                    remaining = total_estimated - elapsed_time
                    
                    self.predictions['completion_time'] = remaining
                else:
                    self.predictions['completion_time'] = timedelta(minutes=30)
            else:
                self.predictions['completion_time'] = timedelta(minutes=20)
                
        except Exception as e:
            print(f"Error prediciendo tiempo de finalización: {e}")
    
    def update_prediction_display(self):
        """Actualiza el display de predicciones"""
        try:
            next_action = self.predictions.get('next_action', 'analyzing')
            probability = self.predictions.get('success_probability', 0.0)
            
            prediction_text = f"🔮 Próxima acción: {next_action} ({probability:.0%})"
            self.prediction_label.configure(text=prediction_text)
            
        except Exception as e:
            print(f"Error actualizando display de predicciones: {e}")
    
    def update_recommendations(self):
        """Actualiza recomendaciones inteligentes"""
        try:
            recommendations = self.generate_intelligent_recommendations()
            
            # Actualizar widget de recomendaciones
            self.recommendations_text.configure(state="normal")
            self.recommendations_text.delete("1.0", "end")
            self.recommendations_text.insert("1.0", recommendations)
            self.recommendations_text.configure(state="disabled")
            
        except Exception as e:
            print(f"Error actualizando recomendaciones: {e}")
    
    def generate_intelligent_recommendations(self) -> str:
        """Genera recomendaciones inteligentes"""
        try:
            current_time = datetime.now()
            cpu_usage = self.system_metrics['cpu_usage']
            memory_usage = self.system_metrics['memory_usage']
            training_progress = self.system_metrics['training_progress']
            
            recommendations = f"""🤖 IA Activa - {current_time.strftime('%H:%M:%S')}

💡 Recomendaciones Inteligentes:
"""
            
            # Recomendaciones basadas en recursos
            if cpu_usage > 80:
                recommendations += "• ⚠️ Alto uso de CPU - Considera reducir batch size\n"
            elif cpu_usage < 30:
                recommendations += "• ⚡ CPU disponible - Puedes aumentar batch size\n"
            
            if memory_usage > 85:
                recommendations += "• 🔴 Memoria alta - Libera recursos no utilizados\n"
            elif memory_usage < 50:
                recommendations += "• 💚 Memoria disponible - Ideal para entrenamientos largos\n"
            
            # Recomendaciones basadas en progreso
            if training_progress > 0:
                recommendations += f"• 📈 Entrenamiento en progreso: {training_progress:.1f}%\n"
                if training_progress > 80:
                    recommendations += "• 🎯 Cerca de completar - Prepara evaluación\n"
            else:
                recommendations += "• 🚀 Sistema listo para iniciar entrenamiento\n"
            
            # Predicciones
            recommendations += f"""
🔮 Predicciones:
• Próxima acción sugerida: {self.predictions.get('next_action', 'Analizando...')}
• Probabilidad de éxito: {self.predictions.get('success_probability', 0):.0%}
"""
            
            if self.predictions.get('completion_time'):
                completion = self.predictions['completion_time']
                if isinstance(completion, timedelta):
                    minutes = int(completion.total_seconds() / 60)
                    recommendations += f"• Tiempo estimado restante: {minutes} minutos\n"
            
            # Acciones sugeridas
            recommendations += """
⚡ Acciones Sugeridas:
"""
            
            if training_progress == 0:
                recommendations += "• Iniciar entrenamiento inteligente\n"
                recommendations += "• Configurar callbacks automáticos\n"
            elif training_progress < 50:
                recommendations += "• Monitorear métricas de validación\n"
                recommendations += "• Ajustar learning rate si es necesario\n"
            else:
                recommendations += "• Preparar evaluación final\n"
                recommendations += "• Considerar early stopping\n"
            
            recommendations += "• Activar auto-optimización\n"
            recommendations += "• Revisar visualizaciones 3D\n"
            
            return recommendations
            
        except Exception as e:
            print(f"Error generando recomendaciones: {e}")
            return "Error generando recomendaciones inteligentes"
    
    def adapt_layout(self):
        """Adapta el layout basado en patrones de uso"""
        try:
            # Adaptar basado en actividad actual
            current_activity = self.user_state['current_activity']
            
            if current_activity == 'training':
                # Priorizar visualización de entrenamiento
                self.viz_notebook.set("Entrenamiento")
            elif current_activity == 'data_analysis':
                # Priorizar visualización de datos
                self.viz_notebook.set("Datos")
            elif current_activity == 'model_design':
                # Priorizar visualización de arquitectura
                self.viz_notebook.set("Arquitectura")
            
        except Exception as e:
            print(f"Error adaptando layout: {e}")
    
    def reorganize_widgets(self):
        """Reorganiza widgets por prioridad de uso"""
        try:
            # Calcular prioridades basadas en uso
            for widget_name in self.adaptive_widgets:
                if widget_name not in self.widget_priorities:
                    self.widget_priorities[widget_name] = 1.0
                
                # Incrementar prioridad si se usa frecuentemente
                # (Implementación simplificada)
                self.widget_priorities[widget_name] *= 0.99  # Decaimiento temporal
            
        except Exception as e:
            print(f"Error reorganizando widgets: {e}")
    
    def adapt_theme(self):
        """Adapta el tema basado en condiciones"""
        try:
            current_hour = datetime.now().hour
            
            # Adaptar colores según hora del día
            if 6 <= current_hour <= 18:
                # Tema diurno (más brillante)
                self.colors['background'] = '#1A1F3A'
                self.colors['surface'] = '#2A2F4A'
            else:
                # Tema nocturno (más oscuro)
                self.colors['background'] = '#0A0E27'
                self.colors['surface'] = '#1A1F3A'
            
        except Exception as e:
            print(f"Error adaptando tema: {e}")
    
    # Métodos de control inteligente
    
    def start_intelligent_training(self):
        """Inicia entrenamiento inteligente"""
        try:
            self.user_state['current_activity'] = 'training'
            self.user_state['interaction_count'] += 1
            
            if self.ai_consciousness:
                response = self.ai_consciousness.process_user_input(
                    "Iniciar entrenamiento inteligente",
                    {'dashboard_state': self.user_state}
                )
                print(f"🧠 IA: {response['text']}")
            
            # Simular inicio de entrenamiento
            self.system_metrics['training_progress'] = 1.0
            
            # Actualizar recomendaciones
            self.update_recommendations()
            
        except Exception as e:
            print(f"Error iniciando entrenamiento inteligente: {e}")
    
    def start_auto_optimization(self):
        """Inicia auto-optimización"""
        try:
            self.user_state['current_activity'] = 'optimization'
            self.user_state['interaction_count'] += 1
            
            if self.ai_consciousness:
                response = self.ai_consciousness.process_user_input(
                    "Iniciar auto-optimización",
                    {'dashboard_state': self.user_state}
                )
                print(f"🧠 IA: {response['text']}")
            
            # Simular optimización
            print("⚡ Auto-optimización iniciada")
            
        except Exception as e:
            print(f"Error iniciando auto-optimización: {e}")
    
    def generate_predictions(self):
        """Genera predicciones manuales"""
        try:
            self.user_state['current_activity'] = 'prediction'
            self.user_state['interaction_count'] += 1
            
            # Forzar actualización de predicciones
            self.update_predictions()
            
            if self.ai_consciousness:
                response = self.ai_consciousness.process_user_input(
                    "Generar predicciones inteligentes",
                    {'predictions': self.predictions}
                )
                print(f"🧠 IA: {response['text']}")
            
        except Exception as e:
            print(f"Error generando predicciones: {e}")
    
    def toggle_auto_learning(self):
        """Activa/desactiva aprendizaje automático"""
        self.dashboard_config['learning_enabled'] = self.auto_learning_switch.get()
        status = "activado" if self.dashboard_config['learning_enabled'] else "desactivado"
        print(f"🧠 Aprendizaje automático {status}")
    
    def toggle_predictions(self):
        """Activa/desactiva predicciones"""
        self.dashboard_config['prediction_enabled'] = self.predictions_switch.get()
        status = "activadas" if self.dashboard_config['prediction_enabled'] else "desactivadas"
        print(f"🔮 Predicciones {status}")
    
    def get_dashboard_state(self) -> Dict:
        """Obtiene el estado actual del dashboard"""
        return {
            'config': self.dashboard_config,
            'user_state': self.user_state,
            'system_metrics': self.system_metrics,
            'predictions': self.predictions,
            'widget_priorities': self.widget_priorities
        }

# Funciones de utilidad para integración

def create_adaptive_dashboard(parent_frame, ai_consciousness=None) -> AdaptiveDashboard:
    """Crea una instancia del dashboard adaptativo"""
    return AdaptiveDashboard(parent_frame, ai_consciousness)

def update_dashboard_metrics(dashboard: AdaptiveDashboard, metrics: Dict):
    """Actualiza métricas del dashboard externamente"""
    dashboard.system_metrics.update(metrics)

def set_dashboard_activity(dashboard: AdaptiveDashboard, activity: str):
    """Establece la actividad actual del usuario"""
    dashboard.user_state['current_activity'] = activity
    dashboard.user_state['last_interaction'] = datetime.now()
    dashboard.user_state['interaction_count'] += 1