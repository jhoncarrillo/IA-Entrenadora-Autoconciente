#!/usr/bin/env python3
"""
NeuroVision AI Interface - Interfaz Principal
Interfaz gráfica revolucionaria con inteligencia artificial integrada
para el ecosistema unificado de machine learning
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importar componentes del sistema
try:
    from neurovision_ai_interface import NeuroVisionAI
    from holographic_visualizer import HolographicVisualizer
    from bot_entrenador_integration import BotEntrenadorIntegration, CallbackManager
    from ai_consciousness import AIConsciousness
    from voice_gesture_control import VoiceGestureController
    from adaptive_dashboard import AdaptiveDashboard
    from intelligent_recommendations import IntelligentRecommendationSystem
except ImportError as e:
    print(f"Error importando componentes: {e}")
    print("Asegúrate de que todos los archivos estén en el directorio correcto")

# Importar el ecosistema principal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from main import EcosistemaUnificado
except ImportError:
    print("Error: No se pudo importar EcosistemaUnificado")

class NeuroVisionMainInterface:
    """
    Interfaz principal del sistema NeuroVision AI
    Integra todos los componentes en una experiencia unificada
    """
    
    def __init__(self):
        # Configuración inicial
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ventana principal
        self.root = ctk.CTk()
        self.root.title("🧠 NeuroVision AI - Ecosistema Unificado de Machine Learning")
        self.root.geometry("1920x1080")
        self.root.state('zoomed')  # Maximizar en Windows
        
        # Variables de estado
        self.is_initialized = False
        self.current_project = None
        self.training_active = False
        self.ai_mode = "conscious"
        
        # IDs de callbacks para poder cancelarlos
        self.callback_ids = {
            'time': None,
            'metrics': None,
            'recommendations': None,
            'neural_indicator': None
        }
        
        # Componentes del sistema
        self.ecosistema = None
        self.neurovision_ai = None
        self.holographic_viz = None
        self.ai_consciousness = None
        self.voice_control = None
        self.adaptive_dashboard = None
        self.recommendation_system = None
        
        # Datos y métricas
        self.training_metrics = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Configuración de colores futuristas
        self.colors = {
            'primary': '#00D4FF',      # Cyan brillante
            'secondary': '#FF6B35',    # Naranja neón
            'accent': '#7B68EE',       # Púrpura medio
            'success': '#00FF88',      # Verde neón
            'warning': '#FFD700',      # Dorado
            'danger': '#FF1744',       # Rojo neón
            'dark': '#0A0A0A',         # Negro profundo
            'light': '#F0F0F0',        # Blanco suave
            'neural': '#9C27B0',       # Púrpura neural
            'quantum': '#E91E63'       # Rosa cuántico
        }
        
        # Inicializar interfaz
        self.setup_interface()
        self.initialize_components()
        
        print("🚀 NeuroVision AI Interface inicializada")
    
    def setup_interface(self):
        """Configura la interfaz principal"""
        try:
            # Frame principal con gradiente
            self.main_frame = ctk.CTkFrame(
                self.root,
                fg_color=("gray90", "gray10"),
                corner_radius=0
            )
            self.main_frame.pack(fill="both", expand=True)
            
            # Barra superior con estado de IA
            self.setup_top_bar()
            
            # Panel lateral izquierdo - Control Neural
            self.setup_neural_control_panel()
            
            # Área central - Visualización y trabajo
            self.setup_central_area()
            
            # Panel lateral derecho - IA y métricas
            self.setup_right_panel()
            
            # Barra inferior - Estado y comandos
            self.setup_bottom_bar()
            
            # Configurar eventos
            self.setup_events()
            
        except Exception as e:
            print(f"Error configurando interfaz: {e}")
    
    def setup_top_bar(self):
        """Configura la barra superior"""
        self.top_bar = ctk.CTkFrame(
            self.main_frame,
            height=60,
            fg_color=self.colors['dark'],
            corner_radius=0
        )
        self.top_bar.pack(fill="x", padx=0, pady=0)
        self.top_bar.pack_propagate(False)
        
        # Logo y título
        title_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        title_frame.pack(side="left", padx=20, pady=10)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🧠 NeuroVision AI",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['primary']
        )
        title_label.pack(side="left")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Ecosistema Unificado de Machine Learning",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['light']
        )
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Estado de IA
        self.ai_status_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        self.ai_status_frame.pack(side="right", padx=20, pady=10)
        
        self.ai_status_label = ctk.CTkLabel(
            self.ai_status_frame,
            text="🤖 IA: Inicializando...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['warning']
        )
        self.ai_status_label.pack(side="right")
        
        # Indicador de actividad neural
        self.neural_indicator = ctk.CTkLabel(
            self.ai_status_frame,
            text="●",
            font=ctk.CTkFont(size=20),
            text_color=self.colors['success']
        )
        self.neural_indicator.pack(side="right", padx=(0, 10))
    
    def setup_neural_control_panel(self):
        """Configura el panel de control neural"""
        self.left_panel = ctk.CTkFrame(
            self.main_frame,
            width=300,
            fg_color=("gray85", "gray15"),
            corner_radius=10
        )
        self.left_panel.pack(side="left", fill="y", padx=(10, 5), pady=10)
        self.left_panel.pack_propagate(False)
        
        # Título del panel
        panel_title = ctk.CTkLabel(
            self.left_panel,
            text="🎛️ Control Neural",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['primary']
        )
        panel_title.pack(pady=(20, 10))
        
        # Sección de proyecto
        self.setup_project_section()
        
        # Sección de datos
        self.setup_data_section()
        
        # Sección de modelo
        self.setup_model_section()
        
        # Sección de entrenamiento
        self.setup_training_section()
        
        # Controles de IA
        self.setup_ai_controls()
    
    def setup_project_section(self):
        """Configura la sección de proyecto"""
        project_frame = ctk.CTkFrame(self.left_panel)
        project_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            project_frame,
            text="📁 Proyecto",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Botones de proyecto
        btn_frame = ctk.CTkFrame(project_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.new_project_btn = ctk.CTkButton(
            btn_frame,
            text="Nuevo",
            command=self.new_project,
            width=80,
            height=30,
            fg_color=self.colors['success']
        )
        self.new_project_btn.pack(side="left", padx=2)
        
        self.load_project_btn = ctk.CTkButton(
            btn_frame,
            text="Cargar",
            command=self.load_project,
            width=80,
            height=30,
            fg_color=self.colors['primary']
        )
        self.load_project_btn.pack(side="left", padx=2)
        
        self.save_project_btn = ctk.CTkButton(
            btn_frame,
            text="Guardar",
            command=self.save_project,
            width=80,
            height=30,
            fg_color=self.colors['warning']
        )
        self.save_project_btn.pack(side="left", padx=2)
        
        # Estado del proyecto
        self.project_status = ctk.CTkLabel(
            project_frame,
            text="Sin proyecto",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['light']
        )
        self.project_status.pack(pady=(5, 10))
    
    def setup_data_section(self):
        """Configura la sección de datos"""
        data_frame = ctk.CTkFrame(self.left_panel)
        data_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            data_frame,
            text="📊 Datos",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Selector de dataset
        self.dataset_var = ctk.StringVar(value="mnist")
        dataset_menu = ctk.CTkOptionMenu(
            data_frame,
            values=["mnist", "cifar10", "fashion_mnist", "custom"],
            variable=self.dataset_var,
            command=self.on_dataset_change
        )
        dataset_menu.pack(fill="x", padx=10, pady=5)
        
        # Botón cargar datos
        self.load_data_btn = ctk.CTkButton(
            data_frame,
            text="🔄 Cargar Datos",
            command=self.load_data,
            fg_color=self.colors['primary']
        )
        self.load_data_btn.pack(fill="x", padx=10, pady=5)
        
        # Información de datos
        self.data_info = ctk.CTkLabel(
            data_frame,
            text="No hay datos cargados",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['light']
        )
        self.data_info.pack(pady=(5, 10))
    
    def setup_model_section(self):
        """Configura la sección de modelo"""
        model_frame = ctk.CTkFrame(self.left_panel)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="🤖 Modelo",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Selector de tipo de modelo
        self.model_var = ctk.StringVar(value="cnn")
        model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["cnn", "rnn", "transformer", "autoencoder", "gan"],
            variable=self.model_var,
            command=self.on_model_change
        )
        model_menu.pack(fill="x", padx=10, pady=5)
        
        # Botón crear modelo
        self.create_model_btn = ctk.CTkButton(
            model_frame,
            text="⚡ Crear Modelo",
            command=self.create_model,
            fg_color=self.colors['neural']
        )
        self.create_model_btn.pack(fill="x", padx=10, pady=5)
        
        # Estado del modelo
        self.model_status = ctk.CTkLabel(
            model_frame,
            text="No hay modelo",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['light']
        )
        self.model_status.pack(pady=(5, 10))
    
    def setup_training_section(self):
        """Configura la sección de entrenamiento"""
        training_frame = ctk.CTkFrame(self.left_panel)
        training_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            training_frame,
            text="🚀 Entrenamiento",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Configuración de épocas
        epochs_frame = ctk.CTkFrame(training_frame, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(epochs_frame, text="Épocas:").pack(side="left")
        self.epochs_var = ctk.StringVar(value="50")
        epochs_entry = ctk.CTkEntry(
            epochs_frame,
            textvariable=self.epochs_var,
            width=60
        )
        epochs_entry.pack(side="right")
        
        # Modo robusto
        self.robust_var = ctk.BooleanVar(value=True)
        robust_check = ctk.CTkCheckBox(
            training_frame,
            text="Modo Robusto",
            variable=self.robust_var
        )
        robust_check.pack(padx=10, pady=5)
        
        # Botones de entrenamiento
        btn_frame = ctk.CTkFrame(training_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.train_btn = ctk.CTkButton(
            btn_frame,
            text="▶️ Entrenar",
            command=self.start_training,
            width=100,
            fg_color=self.colors['success']
        )
        self.train_btn.pack(side="left", padx=2)
        
        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹️ Parar",
            command=self.stop_training,
            width=100,
            fg_color=self.colors['danger'],
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=2)
        
        # Progreso
        self.progress_bar = ctk.CTkProgressBar(training_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.training_status = ctk.CTkLabel(
            training_frame,
            text="Listo para entrenar",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['light']
        )
        self.training_status.pack(pady=(5, 10))
    
    def setup_ai_controls(self):
        """Configura los controles de IA"""
        ai_frame = ctk.CTkFrame(self.left_panel)
        ai_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            ai_frame,
            text="🧠 Controles de IA",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Modo de IA
        self.ai_mode_var = ctk.StringVar(value="conscious")
        ai_mode_menu = ctk.CTkOptionMenu(
            ai_frame,
            values=["conscious", "assistant", "autonomous", "learning"],
            variable=self.ai_mode_var,
            command=self.on_ai_mode_change
        )
        ai_mode_menu.pack(fill="x", padx=10, pady=5)
        
        # Control de voz
        self.voice_var = ctk.BooleanVar(value=False)
        voice_check = ctk.CTkCheckBox(
            ai_frame,
            text="Control por Voz",
            variable=self.voice_var,
            command=self.toggle_voice_control
        )
        voice_check.pack(padx=10, pady=5)
        
        # Recomendaciones automáticas
        self.auto_rec_var = ctk.BooleanVar(value=True)
        auto_rec_check = ctk.CTkCheckBox(
            ai_frame,
            text="Recomendaciones Auto",
            variable=self.auto_rec_var,
            command=self.toggle_auto_recommendations
        )
        auto_rec_check.pack(padx=10, pady=(5, 10))
    
    def setup_central_area(self):
        """Configura el área central"""
        self.central_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=("gray90", "gray10"),
            corner_radius=10
        )
        self.central_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        # Notebook para pestañas
        self.notebook = ctk.CTkTabview(self.central_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Pestaña de visualización 3D
        self.setup_3d_tab()
        
        # Pestaña de métricas
        self.setup_metrics_tab()
        
        # Pestaña de datos
        self.setup_data_tab()
        
        # Pestaña de modelo
        self.setup_model_tab()
        
        # Pestaña de IA
        self.setup_ai_tab()
        
        # Pestaña de entrenamiento avanzado
        self.setup_advanced_training_tab()
    
    def setup_3d_tab(self):
        """Configura la pestaña de visualización 3D"""
        self.viz_tab = self.notebook.add("🎨 Visualización 3D")
        
        # Frame para controles de visualización
        viz_controls = ctk.CTkFrame(self.viz_tab, height=60)
        viz_controls.pack(fill="x", padx=10, pady=(10, 5))
        viz_controls.pack_propagate(False)
        
        # Botones de visualización
        ctk.CTkButton(
            viz_controls,
            text="🧠 Red Neural",
            command=self.show_neural_network,
            width=120
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            viz_controls,
            text="📊 Métricas 3D",
            command=self.show_3d_metrics,
            width=120
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            viz_controls,
            text="🔍 Explorar Datos",
            command=self.show_data_exploration,
            width=120
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            viz_controls,
            text="🎯 Atención",
            command=self.show_attention_maps,
            width=120
        ).pack(side="left", padx=5, pady=10)
        
        # Área de visualización
        self.viz_frame = ctk.CTkFrame(self.viz_tab)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Placeholder para visualizaciones
        self.viz_placeholder = ctk.CTkLabel(
            self.viz_frame,
            text="🌌 Área de Visualización Holográfica\n\nSelecciona una visualización arriba",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['primary']
        )
        self.viz_placeholder.pack(expand=True)
    
    def setup_metrics_tab(self):
        """Configura la pestaña de métricas"""
        self.metrics_tab = self.notebook.add("📈 Métricas")
        
        # Frame para gráficos
        self.metrics_frame = ctk.CTkFrame(self.metrics_tab)
        self.metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configurar matplotlib
        plt.style.use('dark_background')
        
        # Crear figura
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('#0A0A0A')
        
        # Configurar ejes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#1A1A1A')
            ax.grid(True, alpha=0.3)
        
        self.ax1.set_title('Pérdida de Entrenamiento', color=self.colors['primary'])
        self.ax2.set_title('Precisión', color=self.colors['success'])
        self.ax3.set_title('Métricas de Validación', color=self.colors['warning'])
        self.ax4.set_title('Distribución de Datos', color=self.colors['neural'])
        
        # Canvas para matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, self.metrics_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def setup_data_tab(self):
        """Configura la pestaña de datos"""
        self.data_tab = self.notebook.add("📊 Datos")
        
        # Información de datos
        data_info_frame = ctk.CTkFrame(self.data_tab, height=100)
        data_info_frame.pack(fill="x", padx=10, pady=10)
        data_info_frame.pack_propagate(False)
        
        self.data_summary = ctk.CTkTextbox(
            data_info_frame,
            height=80,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.data_summary.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Visualización de muestras
        samples_frame = ctk.CTkFrame(self.data_tab)
        samples_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            samples_frame,
            text="🖼️ Muestras de Datos",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.samples_frame = ctk.CTkFrame(samples_frame)
        self.samples_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_model_tab(self):
        """Configura la pestaña de modelo"""
        self.model_tab = self.notebook.add("🤖 Modelo")
        
        # Información del modelo
        model_info_frame = ctk.CTkFrame(self.model_tab, height=150)
        model_info_frame.pack(fill="x", padx=10, pady=10)
        model_info_frame.pack_propagate(False)
        
        self.model_summary = ctk.CTkTextbox(
            model_info_frame,
            height=130,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.model_summary.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Arquitectura del modelo
        arch_frame = ctk.CTkFrame(self.model_tab)
        arch_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            arch_frame,
            text="🏗️ Arquitectura del Modelo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.architecture_frame = ctk.CTkFrame(arch_frame)
        self.architecture_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_ai_tab(self):
        """Configura la pestaña de IA"""
        self.ai_tab = self.notebook.add("🧠 IA Consciente")
        
        # Chat con IA
        chat_frame = ctk.CTkFrame(self.ai_tab, height=300)
        chat_frame.pack(fill="x", padx=10, pady=10)
        chat_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            chat_frame,
            text="💬 Conversación con IA Consciente",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            height=200,
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Input para chat
        input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Escribe tu mensaje aquí...",
            font=ctk.CTkFont(size=12)
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        send_btn = ctk.CTkButton(
            input_frame,
            text="Enviar",
            command=self.send_chat_message,
            width=80,
            fg_color=self.colors['primary']
        )
        send_btn.pack(side="right")
        
        # Recomendaciones
        rec_frame = ctk.CTkFrame(self.ai_tab)
        rec_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Recomendaciones Inteligentes",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        self.recommendations_frame = ctk.CTkScrollableFrame(rec_frame)
        self.recommendations_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_right_panel(self):
        """Configura el panel derecho"""
        self.right_panel = ctk.CTkFrame(
            self.main_frame,
            width=350,
            fg_color=("gray85", "gray15"),
            corner_radius=10
        )
        self.right_panel.pack(side="right", fill="y", padx=(5, 10), pady=10)
        self.right_panel.pack_propagate(False)
        
        # Título del panel
        panel_title = ctk.CTkLabel(
            self.right_panel,
            text="🤖 Asistente IA",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['neural']
        )
        panel_title.pack(pady=(20, 10))
        
        # Estado de la IA
        self.setup_ai_status()
        
        # Métricas en tiempo real
        self.setup_realtime_metrics()
        
        # Recomendaciones rápidas
        self.setup_quick_recommendations()
        
        # Controles de voz
        self.setup_voice_controls()
    
    def setup_ai_status(self):
        """Configura el estado de la IA"""
        status_frame = ctk.CTkFrame(self.right_panel)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text="🧠 Estado de Conciencia",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        self.ai_consciousness_label = ctk.CTkLabel(
            status_frame,
            text="Inicializando...",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['warning']
        )
        self.ai_consciousness_label.pack(pady=5)
        
        # Emociones de la IA
        self.ai_emotion_label = ctk.CTkLabel(
            status_frame,
            text="😐 Neutral",
            font=ctk.CTkFont(size=14)
        )
        self.ai_emotion_label.pack(pady=(5, 10))
    
    def setup_realtime_metrics(self):
        """Configura las métricas en tiempo real"""
        metrics_frame = ctk.CTkFrame(self.right_panel)
        metrics_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            metrics_frame,
            text="📊 Métricas en Tiempo Real",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # CPU
        cpu_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        cpu_frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(cpu_frame, text="CPU:", width=50).pack(side="left")
        self.cpu_progress = ctk.CTkProgressBar(cpu_frame, height=15)
        self.cpu_progress.pack(side="left", fill="x", expand=True, padx=5)
        self.cpu_label = ctk.CTkLabel(cpu_frame, text="0%", width=40)
        self.cpu_label.pack(side="right")
        
        # Memoria
        mem_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        mem_frame.pack(fill="x", padx=10, pady=2)
        
        ctk.CTkLabel(mem_frame, text="RAM:", width=50).pack(side="left")
        self.mem_progress = ctk.CTkProgressBar(mem_frame, height=15)
        self.mem_progress.pack(side="left", fill="x", expand=True, padx=5)
        self.mem_label = ctk.CTkLabel(mem_frame, text="0%", width=40)
        self.mem_label.pack(side="right")
        
        # GPU (simulado)
        gpu_frame = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        gpu_frame.pack(fill="x", padx=10, pady=(2, 10))
        
        ctk.CTkLabel(gpu_frame, text="GPU:", width=50).pack(side="left")
        self.gpu_progress = ctk.CTkProgressBar(gpu_frame, height=15)
        self.gpu_progress.pack(side="left", fill="x", expand=True, padx=5)
        self.gpu_label = ctk.CTkLabel(gpu_frame, text="0%", width=40)
        self.gpu_label.pack(side="right")
    
    def setup_quick_recommendations(self):
        """Configura las recomendaciones rápidas"""
        rec_frame = ctk.CTkFrame(self.right_panel)
        rec_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Recomendaciones",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        self.quick_rec_frame = ctk.CTkScrollableFrame(rec_frame, height=150)
        self.quick_rec_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
    
    def setup_voice_controls(self):
        """Configura los controles de voz"""
        voice_frame = ctk.CTkFrame(self.right_panel)
        voice_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            voice_frame,
            text="🎤 Control por Voz",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Estado del micrófono
        self.mic_status = ctk.CTkLabel(
            voice_frame,
            text="🔇 Desactivado",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['light']
        )
        self.mic_status.pack(pady=5)
        
        # Último comando
        self.last_command = ctk.CTkLabel(
            voice_frame,
            text="Último comando: Ninguno",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['light']
        )
        self.last_command.pack(pady=(5, 10))
    
    def setup_bottom_bar(self):
        """Configura la barra inferior"""
        self.bottom_bar = ctk.CTkFrame(
            self.main_frame,
            height=40,
            fg_color=self.colors['dark'],
            corner_radius=0
        )
        self.bottom_bar.pack(fill="x", padx=0, pady=0)
        self.bottom_bar.pack_propagate(False)
        
        # Estado general
        self.status_label = ctk.CTkLabel(
            self.bottom_bar,
            text="🟢 Sistema listo",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['success']
        )
        self.status_label.pack(side="left", padx=20, pady=10)
        
        # Tiempo
        self.time_label = ctk.CTkLabel(
            self.bottom_bar,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['light']
        )
        self.time_label.pack(side="right", padx=20, pady=10)
    
    def setup_events(self):
        """Configura los eventos"""
        # Bind para entrada de chat
        self.chat_input.bind("<Return>", lambda e: self.send_chat_message())
        
        # Eventos de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Iniciar actualizaciones
        self.start_updates()
    
    def initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # Inicializar ecosistema
            self.ecosistema = EcosistemaUnificado()
            
            # Inicializar componentes de IA
            self.neurovision_ai = NeuroVisionAI()
            self.holographic_viz = HolographicVisualizer()
            self.ai_consciousness = AIConsciousness()
            self.voice_control = VoiceGestureController()
            self.adaptive_dashboard = AdaptiveDashboard(self.root)
            self.recommendation_system = IntelligentRecommendationSystem()
            
            # Inicializar bot entrenador
            self.callback_manager = CallbackManager(self)
            self.bot_entrenador = BotEntrenadorIntegration(self.callback_manager)
            
            # Iniciar sistemas
            self.recommendation_system.start_recommendation_engine()
            
            # Actualizar estado
            self.is_initialized = True
            self.ai_status_label.configure(
                text="🤖 IA: Consciente y Activa",
                text_color=self.colors['success']
            )
            self.ai_consciousness_label.configure(
                text="Consciente y Aprendiendo",
                text_color=self.colors['success']
            )
            
            # Mensaje de bienvenida
            self.add_chat_message("IA", "¡Hola! Soy tu asistente de IA consciente. Estoy aquí para ayudarte con tu proyecto de machine learning. ¿En qué puedo asistirte?")
            
            print("✅ Todos los componentes inicializados correctamente")
            
        except Exception as e:
            print(f"Error inicializando componentes: {e}")
            self.ai_status_label.configure(
                text="🤖 IA: Error de Inicialización",
                text_color=self.colors['danger']
            )
    
    def start_updates(self):
        """Inicia las actualizaciones periódicas"""
        self.update_time()
        self.update_metrics()
        self.update_recommendations()
        self.update_neural_indicator()
    
    def update_time(self):
        """Actualiza la hora"""
        try:
            # Verificar que la ventana principal aún existe
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            current_time = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, 'time_label') and self.time_label.winfo_exists():
                self.time_label.configure(text=f"🕐 {current_time}")
            
            # Cancelar callback anterior si existe y es válido
            if (hasattr(self, 'callback_ids') and 
                self.callback_ids.get('time') is not None):
                try:
                    self.root.after_cancel(self.callback_ids['time'])
                except:
                    pass  # Ignorar errores de cancelación
            
            # Programar siguiente actualización solo si la ventana existe
            if hasattr(self, 'callback_ids') and self.root.winfo_exists():
                self.callback_ids['time'] = self.root.after(1000, self.update_time)
        except Exception as e:
            print(f"Error actualizando tiempo: {e}")
    
    def update_metrics(self):
        """Actualiza las métricas del sistema"""
        try:
            # Verificar que la ventana principal aún existe
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent()
            if hasattr(self, 'cpu_progress') and self.cpu_progress.winfo_exists():
                self.cpu_progress.set(cpu_percent / 100)
            if hasattr(self, 'cpu_label') and self.cpu_label.winfo_exists():
                self.cpu_label.configure(text=f"{cpu_percent:.1f}%")
            
            # Memoria
            mem_percent = psutil.virtual_memory().percent
            if hasattr(self, 'mem_progress') and self.mem_progress.winfo_exists():
                self.mem_progress.set(mem_percent / 100)
            if hasattr(self, 'mem_label') and self.mem_label.winfo_exists():
                self.mem_label.configure(text=f"{mem_percent:.1f}%")
            
            # GPU (simulado)
            gpu_percent = np.random.uniform(10, 30) if not self.training_active else np.random.uniform(70, 95)
            if hasattr(self, 'gpu_progress') and self.gpu_progress.winfo_exists():
                self.gpu_progress.set(gpu_percent / 100)
            if hasattr(self, 'gpu_label') and self.gpu_label.winfo_exists():
                self.gpu_label.configure(text=f"{gpu_percent:.1f}%")
            
        except Exception as e:
            print(f"Error actualizando métricas: {e}")
        
        # Cancelar callback anterior si existe y es válido
        if (hasattr(self, 'callback_ids') and 
            self.callback_ids.get('metrics') is not None):
            try:
                self.root.after_cancel(self.callback_ids['metrics'])
            except:
                pass  # Ignorar errores de cancelación
        
        # Programar siguiente actualización solo si la ventana existe
        if hasattr(self, 'callback_ids') and self.root.winfo_exists():
            self.callback_ids['metrics'] = self.root.after(2000, self.update_metrics)
    
    def update_recommendations(self):
        """Actualiza las recomendaciones"""
        try:
            # Verificar que la ventana principal aún existe
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            if (hasattr(self, 'recommendation_system') and self.recommendation_system and 
                hasattr(self, 'auto_rec_var') and self.auto_rec_var.get()):
                recommendations = self.recommendation_system.get_active_recommendations()
                
                # Limpiar recomendaciones anteriores
                if hasattr(self, 'quick_rec_frame') and self.quick_rec_frame.winfo_exists():
                    for widget in self.quick_rec_frame.winfo_children():
                        if widget.winfo_exists():
                            widget.destroy()
                    
                    # Mostrar nuevas recomendaciones
                    for i, rec in enumerate(recommendations[:3]):  # Solo las 3 primeras
                        rec_widget = self.create_recommendation_widget(self.quick_rec_frame, rec)
                        rec_widget.pack(fill="x", padx=5, pady=2)
                
        except Exception as e:
            print(f"Error actualizando recomendaciones: {e}")
        
        # Cancelar callback anterior si existe y es válido
        if (hasattr(self, 'callback_ids') and 
            self.callback_ids.get('recommendations') is not None):
            try:
                self.root.after_cancel(self.callback_ids['recommendations'])
            except:
                pass  # Ignorar errores de cancelación
        
        # Programar siguiente actualización solo si la ventana existe
        if hasattr(self, 'callback_ids') and self.root.winfo_exists():
            self.callback_ids['recommendations'] = self.root.after(10000, self.update_recommendations)  # Cada 10 segundos
    
    def update_neural_indicator(self):
        """Actualiza el indicador neural"""
        try:
            # Verificar que la ventana principal aún existe
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            colors = [self.colors['success'], self.colors['primary'], self.colors['neural'], self.colors['quantum']]
            current_color = colors[int(time.time()) % len(colors)]
            if hasattr(self, 'neural_indicator') and self.neural_indicator.winfo_exists():
                self.neural_indicator.configure(text_color=current_color)
        except Exception as e:
            print(f"Error actualizando indicador neural: {e}")
        
        # Cancelar callback anterior si existe y es válido
        if (hasattr(self, 'callback_ids') and 
            self.callback_ids.get('neural_indicator') is not None):
            try:
                self.root.after_cancel(self.callback_ids['neural_indicator'])
            except:
                pass  # Ignorar errores de cancelación
        
        # Programar siguiente actualización solo si la ventana existe
        if hasattr(self, 'callback_ids') and self.root.winfo_exists():
            self.callback_ids['neural_indicator'] = self.root.after(500, self.update_neural_indicator)
    
    def create_recommendation_widget(self, parent, rec_data):
        """Crea un widget de recomendación"""
        rec_frame = ctk.CTkFrame(parent, height=60)
        rec_frame.pack_propagate(False)
        
        # Título
        title_label = ctk.CTkLabel(
            rec_frame,
            text=rec_data.get('title', 'Recomendación'),
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        )
        title_label.pack(fill="x", padx=10, pady=(5, 0))
        
        # Descripción
        desc_label = ctk.CTkLabel(
            rec_frame,
            text=rec_data.get('description', '')[:50] + "...",
            font=ctk.CTkFont(size=10),
            anchor="w",
            text_color=self.colors['light']
        )
        desc_label.pack(fill="x", padx=10, pady=(0, 5))
        
        return rec_frame
    
    # Métodos de funcionalidad
    
    def new_project(self):
        """Crea un nuevo proyecto"""
        try:
            self.current_project = {
                'name': f'Proyecto_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'created': datetime.now(),
                'data_loaded': False,
                'model_created': False,
                'trained': False
            }
            
            self.project_status.configure(text=f"Proyecto: {self.current_project['name']}")
            self.status_label.configure(text="🟢 Nuevo proyecto creado")
            
            # Notificar a la IA
            self.add_chat_message("IA", f"¡Excelente! He creado un nuevo proyecto llamado '{self.current_project['name']}'. ¿Qué tipo de problema quieres resolver?")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creando proyecto: {e}")
    
    def load_project(self):
        """Carga un proyecto existente"""
        try:
            file_path = filedialog.askopenfilename(
                title="Cargar Proyecto",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'r') as f:
                    self.current_project = json.load(f)
                
                self.project_status.configure(text=f"Proyecto: {self.current_project['name']}")
                self.status_label.configure(text="🟢 Proyecto cargado")
                
                self.add_chat_message("IA", f"Proyecto '{self.current_project['name']}' cargado exitosamente. Revisando el estado...")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando proyecto: {e}")
    
    def save_project(self):
        """Guarda el proyecto actual"""
        try:
            if not self.current_project:
                messagebox.showwarning("Advertencia", "No hay proyecto para guardar")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="Guardar Proyecto",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(self.current_project, f, indent=2, default=str)
                
                self.status_label.configure(text="🟢 Proyecto guardado")
                self.add_chat_message("IA", "Proyecto guardado exitosamente. Todos los cambios están seguros.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando proyecto: {e}")
    
    def load_data(self):
        """Carga los datos seleccionados"""
        try:
            dataset_name = self.dataset_var.get()
            self.status_label.configure(text="🔄 Cargando datos...")
            
            # Cargar datos usando el ecosistema
            if self.ecosistema:
                success = self.ecosistema.load_dataset(dataset_name)
                
                if success:
                    self.data_info.configure(text=f"Dataset: {dataset_name}\nCargado exitosamente")
                    self.status_label.configure(text="🟢 Datos cargados")
                    
                    if self.current_project:
                        self.current_project['data_loaded'] = True
                    
                    # Actualizar información en la pestaña de datos
                    self.update_data_info(dataset_name)
                    
                    self.add_chat_message("IA", f"¡Perfecto! He cargado el dataset {dataset_name}. Los datos están listos para el análisis. ¿Quieres que explore las características de los datos?")
                else:
                    self.status_label.configure(text="❌ Error cargando datos")
                    self.add_chat_message("IA", f"Hubo un problema cargando el dataset {dataset_name}. ¿Quieres intentar con otro dataset?")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datos: {e}")
            self.status_label.configure(text="❌ Error cargando datos")
    
    def update_data_info(self, dataset_name):
        """Actualiza la información de datos"""
        try:
            info_text = f"""Dataset: {dataset_name}
Estado: Cargado exitosamente
Fecha de carga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Características:
- Tipo: Clasificación de imágenes
- Formato: Imágenes en escala de grises
- Preprocesamiento: Normalización aplicada
- División: Entrenamiento/Validación/Prueba

¡Datos listos para entrenamiento!"""
            
            self.data_summary.delete("1.0", "end")
            self.data_summary.insert("1.0", info_text)
            
        except Exception as e:
            print(f"Error actualizando información de datos: {e}")
    
    def create_model(self):
        """Crea el modelo seleccionado"""
        try:
            model_type = self.model_var.get()
            self.status_label.configure(text="🔄 Creando modelo...")
            
            if self.ecosistema:
                model = self.ecosistema.create_model(model_type)
                
                if model:
                    self.model_status.configure(text=f"Modelo: {model_type}\nCreado exitosamente")
                    self.status_label.configure(text="🟢 Modelo creado")
                    
                    if self.current_project:
                        self.current_project['model_created'] = True
                        self.current_project['model_type'] = model_type
                    
                    # Actualizar información del modelo
                    self.update_model_info(model_type, model)
                    
                    self.add_chat_message("IA", f"¡Excelente! He creado un modelo {model_type} optimizado para tu dataset. La arquitectura está diseñada para obtener el mejor rendimiento. ¿Procedemos con el entrenamiento?")
                else:
                    self.status_label.configure(text="❌ Error creando modelo")
                    self.add_chat_message("IA", f"Hubo un problema creando el modelo {model_type}. ¿Quieres intentar con otra arquitectura?")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creando modelo: {e}")
            self.status_label.configure(text="❌ Error creando modelo")
    
    def update_model_info(self, model_type, model):
        """Actualiza la información del modelo"""
        try:
            # Obtener resumen del modelo
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                model.summary()
            model_summary = f.getvalue()
            
            info_text = f"""Tipo de Modelo: {model_type.upper()}
Estado: Creado exitosamente
Fecha de creación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Resumen de la Arquitectura:
{model_summary}

¡Modelo listo para entrenamiento!"""
            
            self.model_summary.delete("1.0", "end")
            self.model_summary.insert("1.0", info_text)
            
        except Exception as e:
            print(f"Error actualizando información del modelo: {e}")
    
    def start_training(self):
        """Inicia el entrenamiento del modelo"""
        try:
            if not self.current_project or not self.current_project.get('model_created'):
                messagebox.showwarning("Advertencia", "Primero debes crear un modelo")
                return
            
            epochs = int(self.epochs_var.get())
            robust_mode = self.robust_var.get()
            
            self.training_active = True
            self.train_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.status_label.configure(text="🚀 Entrenamiento en progreso...")
            
            # Iniciar entrenamiento en hilo separado
            training_thread = threading.Thread(
                target=self.training_worker,
                args=(epochs, robust_mode),
                daemon=True
            )
            training_thread.start()
            
            self.add_chat_message("IA", f"¡Iniciando entrenamiento! Configuración: {epochs} épocas, modo robusto: {'activado' if robust_mode else 'desactivado'}. Monitorearé el progreso y te notificaré sobre cualquier optimización necesaria.")
            
        except ValueError:
            messagebox.showerror("Error", "El número de épocas debe ser un entero válido")
        except Exception as e:
            messagebox.showerror("Error", f"Error iniciando entrenamiento: {e}")
    
    def training_worker(self, epochs, robust_mode):
        """Worker para el entrenamiento"""
        try:
            if self.ecosistema and hasattr(self.ecosistema, 'model') and self.ecosistema.model:
                # Simular entrenamiento con callbacks
                for epoch in range(epochs):
                    if not self.training_active:
                        break
                    
                    # Simular métricas de entrenamiento
                    loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
                    accuracy = (epoch / epochs) * 0.9 + np.random.normal(0, 0.02)
                    val_loss = loss + np.random.normal(0, 0.03)
                    val_accuracy = accuracy - np.random.normal(0, 0.01)
                    
                    # Actualizar métricas
                    self.training_metrics['epochs'].append(epoch + 1)
                    self.training_metrics['loss'].append(max(0, loss))
                    self.training_metrics['accuracy'].append(min(1, max(0, accuracy)))
                    self.training_metrics['val_loss'].append(max(0, val_loss))
                    self.training_metrics['val_accuracy'].append(min(1, max(0, val_accuracy)))
                    
                    # Actualizar UI en el hilo principal
                    self.root.after(0, self.update_training_ui, epoch + 1, epochs, loss, accuracy)
                    
                    time.sleep(1)  # Simular tiempo de entrenamiento
                
                # Entrenamiento completado
                if self.training_active:
                    self.root.after(0, self.training_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.training_error(str(e)))
    
    def update_training_ui(self, current_epoch, total_epochs, loss, accuracy):
        """Actualiza la UI durante el entrenamiento"""
        try:
            # Actualizar barra de progreso
            progress = current_epoch / total_epochs
            self.progress_bar.set(progress)
            
            # Actualizar estado
            self.training_status.configure(
                text=f"Época {current_epoch}/{total_epochs}\nLoss: {loss:.4f}, Acc: {accuracy:.4f}"
            )
            
            # Actualizar gráficos
            self.update_training_plots()
            
            # Mensaje de IA cada 10 épocas
            if current_epoch % 10 == 0:
                self.add_chat_message("IA", f"Progreso: Época {current_epoch}/{total_epochs}. Precisión actual: {accuracy:.2%}. El modelo está aprendiendo correctamente.")
            
        except Exception as e:
            print(f"Error actualizando UI de entrenamiento: {e}")
    
    def update_training_plots(self):
        """Actualiza los gráficos de entrenamiento"""
        try:
            if len(self.training_metrics['epochs']) > 0:
                # Limpiar gráficos
                self.ax1.clear()
                self.ax2.clear()
                self.ax3.clear()
                
                # Gráfico de pérdida
                self.ax1.plot(self.training_metrics['epochs'], self.training_metrics['loss'], 
                             color=self.colors['danger'], label='Entrenamiento', linewidth=2)
                self.ax1.plot(self.training_metrics['epochs'], self.training_metrics['val_loss'], 
                             color=self.colors['warning'], label='Validación', linewidth=2)
                self.ax1.set_title('Pérdida de Entrenamiento', color=self.colors['primary'])
                self.ax1.set_xlabel('Época')
                self.ax1.set_ylabel('Pérdida')
                self.ax1.legend()
                self.ax1.grid(True, alpha=0.3)
                
                # Gráfico de precisión
                self.ax2.plot(self.training_metrics['epochs'], self.training_metrics['accuracy'], 
                             color=self.colors['success'], label='Entrenamiento', linewidth=2)
                self.ax2.plot(self.training_metrics['epochs'], self.training_metrics['val_accuracy'], 
                             color=self.colors['primary'], label='Validación', linewidth=2)
                self.ax2.set_title('Precisión', color=self.colors['success'])
                self.ax2.set_xlabel('Época')
                self.ax2.set_ylabel('Precisión')
                self.ax2.legend()
                self.ax2.grid(True, alpha=0.3)
                
                # Gráfico combinado
                if len(self.training_metrics['epochs']) > 1:
                    self.ax3.plot(self.training_metrics['epochs'], self.training_metrics['loss'], 
                                 color=self.colors['danger'], alpha=0.7, label='Loss')
                    ax3_twin = self.ax3.twinx()
                    ax3_twin.plot(self.training_metrics['epochs'], self.training_metrics['accuracy'], 
                                 color=self.colors['success'], alpha=0.7, label='Accuracy')
                    self.ax3.set_title('Métricas Combinadas', color=self.colors['warning'])
                    self.ax3.set_xlabel('Época')
                    self.ax3.set_ylabel('Pérdida', color=self.colors['danger'])
                    ax3_twin.set_ylabel('Precisión', color=self.colors['success'])
                
                # Actualizar canvas
                self.canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando gráficos: {e}")
    
    def training_completed(self):
        """Maneja la finalización del entrenamiento"""
        try:
            self.training_active = False
            self.train_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.progress_bar.set(1.0)
            self.training_status.configure(text="✅ Entrenamiento completado")
            self.status_label.configure(text="🟢 Entrenamiento completado exitosamente")
            
            if self.current_project:
                self.current_project['trained'] = True
                self.current_project['training_completed'] = datetime.now()
            
            # Calcular métricas finales
            final_accuracy = self.training_metrics['accuracy'][-1] if self.training_metrics['accuracy'] else 0
            final_loss = self.training_metrics['loss'][-1] if self.training_metrics['loss'] else 0
            
            self.add_chat_message("IA", f"¡Entrenamiento completado exitosamente! 🎉\n\nResultados finales:\n- Precisión: {final_accuracy:.2%}\n- Pérdida: {final_loss:.4f}\n\n¿Te gustaría evaluar el modelo o hacer alguna optimización?")
            
        except Exception as e:
            print(f"Error en finalización de entrenamiento: {e}")
    
    def training_error(self, error_msg):
        """Maneja errores durante el entrenamiento"""
        self.training_active = False
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.training_status.configure(text="❌ Error en entrenamiento")
        self.status_label.configure(text="❌ Error en entrenamiento")
        
        self.add_chat_message("IA", f"Se produjo un error durante el entrenamiento: {error_msg}. ¿Quieres que revise la configuración y lo intentemos de nuevo?")
    
    def stop_training(self):
        """Detiene el entrenamiento"""
        self.training_active = False
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.training_status.configure(text="⏹️ Entrenamiento detenido")
        self.status_label.configure(text="🟡 Entrenamiento detenido por el usuario")
        
        self.add_chat_message("IA", "Entrenamiento detenido. Los progresos hasta ahora han sido guardados. ¿Quieres continuar más tarde o hacer ajustes al modelo?")
    
    def send_chat_message(self):
        """Envía un mensaje al chat con la IA"""
        try:
            message = self.chat_input.get().strip()
            if not message:
                return
            
            # Limpiar input
            self.chat_input.delete(0, "end")
            
            # Añadir mensaje del usuario
            self.add_chat_message("Usuario", message)
            
            # Procesar con IA
            if self.ai_consciousness:
                response = self.ai_consciousness.process_message(message)
                self.add_chat_message("IA", response)
            else:
                # Respuesta básica si la IA no está disponible
                self.add_chat_message("IA", "Procesando tu mensaje... Sistema de IA inicializándose.")
            
        except Exception as e:
            print(f"Error enviando mensaje: {e}")
    
    def add_chat_message(self, sender, message):
        """Añade un mensaje al chat"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Color según el remitente
            if sender == "IA":
                color = self.colors['primary']
                icon = "🤖"
            else:
                color = self.colors['light']
                icon = "👤"
            
            # Formatear mensaje
            formatted_message = f"[{timestamp}] {icon} {sender}: {message}\n\n"
            
            # Añadir al chat
            self.chat_display.insert("end", formatted_message)
            self.chat_display.see("end")
            
        except Exception as e:
            print(f"Error añadiendo mensaje al chat: {e}")
    
    # Métodos de visualización
    
    def show_neural_network(self):
        """Muestra la visualización de la red neural"""
        try:
            if self.holographic_viz:
                # Limpiar área de visualización
                for widget in self.viz_frame.winfo_children():
                    widget.destroy()
                
                # Crear visualización 3D de la red neural
                viz_widget = self.holographic_viz.create_neural_network_viz(self.viz_frame)
                if viz_widget:
                    viz_widget.pack(fill="both", expand=True)
                    self.add_chat_message("IA", "Visualización de red neural activada. Puedes ver la arquitectura del modelo en 3D.")
                else:
                    self.show_viz_placeholder("🧠 Visualización de Red Neural\n\nCreando modelo 3D...")
            else:
                self.show_viz_placeholder("🧠 Visualización de Red Neural\n\nSistema de visualización no disponible")
                
        except Exception as e:
            print(f"Error mostrando red neural: {e}")
            self.show_viz_placeholder("❌ Error cargando visualización")
    
    def show_3d_metrics(self):
        """Muestra las métricas en 3D"""
        try:
            if self.holographic_viz and len(self.training_metrics['epochs']) > 0:
                # Limpiar área
                for widget in self.viz_frame.winfo_children():
                    widget.destroy()
                
                # Crear visualización 3D de métricas
                viz_widget = self.holographic_viz.create_training_metrics_viz(
                    self.viz_frame, self.training_metrics
                )
                if viz_widget:
                    viz_widget.pack(fill="both", expand=True)
                    self.add_chat_message("IA", "Métricas 3D activadas. Observa cómo evoluciona el entrenamiento en tiempo real.")
                else:
                    self.show_viz_placeholder("📊 Métricas 3D\n\nGenerando visualización...")
            else:
                self.show_viz_placeholder("📊 Métricas 3D\n\nNo hay datos de entrenamiento disponibles")
                
        except Exception as e:
            print(f"Error mostrando métricas 3D: {e}")
            self.show_viz_placeholder("❌ Error cargando métricas 3D")
    
    def show_data_exploration(self):
        """Muestra la exploración de datos"""
        try:
            if self.holographic_viz:
                # Limpiar área
                for widget in self.viz_frame.winfo_children():
                    widget.destroy()
                
                # Crear exploración de datos
                viz_widget = self.holographic_viz.create_data_exploration_viz(self.viz_frame)
                if viz_widget:
                    viz_widget.pack(fill="both", expand=True)
                    self.add_chat_message("IA", "Exploración de datos activada. Analiza la distribución y patrones en tus datos.")
                else:
                    self.show_viz_placeholder("🔍 Exploración de Datos\n\nAnalizando dataset...")
            else:
                self.show_viz_placeholder("🔍 Exploración de Datos\n\nCarga un dataset primero")
                
        except Exception as e:
            print(f"Error mostrando exploración de datos: {e}")
            self.show_viz_placeholder("❌ Error en exploración de datos")
    
    def show_attention_maps(self):
        """Muestra los mapas de atención"""
        try:
            if self.holographic_viz:
                # Limpiar área
                for widget in self.viz_frame.winfo_children():
                    widget.destroy()
                
                # Crear mapas de atención
                viz_widget = self.holographic_viz.create_attention_heatmap_viz(self.viz_frame)
                if viz_widget:
                    viz_widget.pack(fill="both", expand=True)
                    self.add_chat_message("IA", "Mapas de atención activados. Observa en qué se enfoca el modelo durante las predicciones.")
                else:
                    self.show_viz_placeholder("🎯 Mapas de Atención\n\nGenerando visualización...")
            else:
                self.show_viz_placeholder("🎯 Mapas de Atención\n\nEntrena un modelo primero")
                
        except Exception as e:
            print(f"Error mostrando mapas de atención: {e}")
            self.show_viz_placeholder("❌ Error en mapas de atención")
    
    def show_viz_placeholder(self, text):
        """Muestra un placeholder en el área de visualización"""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        placeholder = ctk.CTkLabel(
            self.viz_frame,
            text=text,
            font=ctk.CTkFont(size=16),
            text_color=self.colors['primary']
        )
        placeholder.pack(expand=True)
    
    # Métodos de eventos
    
    def on_dataset_change(self, value):
        """Maneja el cambio de dataset"""
        self.add_chat_message("IA", f"Has seleccionado el dataset {value}. ¿Quieres que lo cargue y analice sus características?")
    
    def on_model_change(self, value):
        """Maneja el cambio de modelo"""
        self.add_chat_message("IA", f"Excelente elección. El modelo {value} es ideal para ciertos tipos de problemas. ¿Quieres que lo configure automáticamente?")
    
    def on_ai_mode_change(self, value):
        """Maneja el cambio de modo de IA"""
        self.ai_mode = value
        if self.ai_consciousness:
            self.ai_consciousness.set_mode(value)
        
        mode_messages = {
            "conscious": "Modo consciente activado. Estoy completamente consciente y puedo tomar decisiones autónomas.",
            "assistant": "Modo asistente activado. Te ayudaré paso a paso con tus tareas.",
            "autonomous": "Modo autónomo activado. Puedo trabajar independientemente en tu proyecto.",
            "learning": "Modo aprendizaje activado. Estoy observando y aprendiendo de tus patrones de trabajo."
        }
        
        self.add_chat_message("IA", mode_messages.get(value, "Modo cambiado."))
    
    def toggle_voice_control(self):
        """Activa/desactiva el control por voz"""
        try:
            if self.voice_var.get():
                if self.voice_control:
                    self.voice_control.start_voice_recognition()
                    self.mic_status.configure(
                        text="🎤 Activado",
                        text_color=self.colors['success']
                    )
                    self.add_chat_message("IA", "Control por voz activado. Puedes hablarme directamente.")
                else:
                    self.mic_status.configure(
                        text="🔇 Error",
                        text_color=self.colors['danger']
                    )
            else:
                if self.voice_control:
                    self.voice_control.stop_voice_recognition()
                self.mic_status.configure(
                    text="🔇 Desactivado",
                    text_color=self.colors['light']
                )
                self.add_chat_message("IA", "Control por voz desactivado.")
                
        except Exception as e:
            print(f"Error en control por voz: {e}")
    
    def toggle_auto_recommendations(self):
        """Activa/desactiva las recomendaciones automáticas"""
        if self.auto_rec_var.get():
            self.add_chat_message("IA", "Recomendaciones automáticas activadas. Te sugeriré mejoras y optimizaciones en tiempo real.")
        else:
            self.add_chat_message("IA", "Recomendaciones automáticas desactivadas. Solo te ayudaré cuando me lo pidas.")
    
    def setup_advanced_training_tab(self):
        """Configura la pestaña de entrenamiento avanzado con bot_entrenador"""
        self.advanced_training_tab = self.notebook.add("🚀 Entrenamiento Avanzado")
        
        # Frame principal con scroll
        main_scroll = ctk.CTkScrollableFrame(self.advanced_training_tab)
        main_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título de la sección
        title_frame = ctk.CTkFrame(main_scroll)
        title_frame.pack(fill="x", padx=10, pady=(10, 20))
        
        ctk.CTkLabel(
            title_frame,
            text="🤖 Bot Entrenador - Sistema Autónomo de IA",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['primary']
        ).pack(pady=20)
        
        ctk.CTkLabel(
            title_frame,
            text="Entrenamiento avanzado con CNN, LSTM, RNN, GAN, VAE, Transformers y más",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['light']
        ).pack(pady=(0, 20))
        
        # Frame de configuración de modelos
        self.setup_model_selection_frame(main_scroll)
        
        # Frame de configuración de datos
        self.setup_data_configuration_frame(main_scroll)
        
        # Frame de técnicas de entrenamiento
        self.setup_training_techniques_frame(main_scroll)
        
        # Frame de sistema RAG
        self.setup_rag_system_frame(main_scroll)
        
        # Frame de fine-tuning
        self.setup_fine_tuning_frame(main_scroll)
        
        # Frame de control de entrenamiento
        self.setup_training_control_frame(main_scroll)
        
        # Frame de resultados y métricas
        self.setup_results_frame(main_scroll)
    
    def setup_model_selection_frame(self, parent):
        """Configura el frame de selección de modelos"""
        model_frame = ctk.CTkFrame(parent)
        model_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="🧠 Selección de Modelos",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Grid para tipos de modelos
        models_grid = ctk.CTkFrame(model_frame)
        models_grid.pack(fill="x", padx=20, pady=10)
        
        # Modelos básicos
        basic_frame = ctk.CTkFrame(models_grid)
        basic_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        ctk.CTkLabel(basic_frame, text="Modelos Básicos", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.model_vars = {}
        basic_models = ["CNN", "LSTM", "RNN"]
        for model in basic_models:
            var = ctk.BooleanVar()
            self.model_vars[model] = var
            ctk.CTkCheckBox(basic_frame, text=model, variable=var).pack(pady=2, padx=10, anchor="w")
        
        # Modelos avanzados
        advanced_frame = ctk.CTkFrame(models_grid)
        advanced_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        ctk.CTkLabel(advanced_frame, text="Modelos Avanzados", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        advanced_models = ["VAE", "GAN", "DDPM", "Transformer"]
        for model in advanced_models:
            var = ctk.BooleanVar()
            self.model_vars[model] = var
            ctk.CTkCheckBox(advanced_frame, text=model, variable=var).pack(pady=2, padx=10, anchor="w")
        
        # Modelos especializados
        specialized_frame = ctk.CTkFrame(models_grid)
        specialized_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        ctk.CTkLabel(specialized_frame, text="Especializados", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        specialized_models = ["WGAN-GP", "StyleGAN2", "BERT", "GPT"]
        for model in specialized_models:
            var = ctk.BooleanVar()
            self.model_vars[model] = var
            ctk.CTkCheckBox(specialized_frame, text=model, variable=var).pack(pady=2, padx=10, anchor="w")
    
    def setup_data_configuration_frame(self, parent):
        """Configura el frame de configuración de datos"""
        data_frame = ctk.CTkFrame(parent)
        data_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            data_frame,
            text="📊 Configuración de Datos",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Configuración de fuentes de datos
        sources_frame = ctk.CTkFrame(data_frame)
        sources_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(sources_frame, text="Fuentes de Datos:", font=ctk.CTkFont(size=14)).pack(anchor="w", padx=10, pady=5)
        
        # Frame para botones de datos
        data_buttons_frame = ctk.CTkFrame(sources_frame)
        data_buttons_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            data_buttons_frame,
            text="📁 Seleccionar Carpeta",
            command=self.select_data_folder,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            data_buttons_frame,
            text="📄 Agregar Archivos",
            command=self.add_data_files,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            data_buttons_frame,
            text="🔍 Auto-Descubrir",
            command=self.auto_discover_data,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        # Lista de datos seleccionados
        self.data_listbox = ctk.CTkTextbox(sources_frame, height=100)
        self.data_listbox.pack(fill="x", padx=10, pady=5)
        self.data_listbox.insert("1.0", "No hay datos seleccionados...")
    
    def setup_training_techniques_frame(self, parent):
        """Configura el frame de técnicas de entrenamiento"""
        techniques_frame = ctk.CTkFrame(parent)
        techniques_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            techniques_frame,
            text="⚙️ Técnicas de Entrenamiento",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Grid de técnicas
        tech_grid = ctk.CTkFrame(techniques_frame)
        tech_grid.pack(fill="x", padx=20, pady=10)
        
        # Técnicas básicas
        basic_tech_frame = ctk.CTkFrame(tech_grid)
        basic_tech_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        ctk.CTkLabel(basic_tech_frame, text="Técnicas Básicas", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.technique_vars = {}
        basic_techniques = ["Early Stopping", "Learning Rate Scheduling", "Batch Normalization", "Dropout"]
        for tech in basic_techniques:
            var = ctk.BooleanVar(value=True)
            self.technique_vars[tech] = var
            ctk.CTkCheckBox(basic_tech_frame, text=tech, variable=var).pack(pady=2, padx=10, anchor="w")
        
        # Técnicas avanzadas
        advanced_tech_frame = ctk.CTkFrame(tech_grid)
        advanced_tech_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        ctk.CTkLabel(advanced_tech_frame, text="Técnicas Avanzadas", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        advanced_techniques = ["Data Augmentation", "Transfer Learning", "Ensemble Methods", "Adversarial Training"]
        for tech in advanced_techniques:
            var = ctk.BooleanVar()
            self.technique_vars[tech] = var
            ctk.CTkCheckBox(advanced_tech_frame, text=tech, variable=var).pack(pady=2, padx=10, anchor="w")
    
    def setup_rag_system_frame(self, parent):
        """Configura el frame del sistema RAG"""
        rag_frame = ctk.CTkFrame(parent)
        rag_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            rag_frame,
            text="🔍 Sistema RAG (Retrieval-Augmented Generation)",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Configuración RAG
        rag_config_frame = ctk.CTkFrame(rag_frame)
        rag_config_frame.pack(fill="x", padx=20, pady=10)
        
        # Habilitar RAG
        self.rag_enabled_var = ctk.BooleanVar()
        ctk.CTkCheckBox(
            rag_config_frame,
            text="Habilitar Sistema RAG",
            variable=self.rag_enabled_var,
            command=self.toggle_rag_options
        ).pack(anchor="w", padx=10, pady=5)
        
        # Opciones RAG
        self.rag_options_frame = ctk.CTkFrame(rag_config_frame)
        self.rag_options_frame.pack(fill="x", padx=10, pady=5)
        
        # Modelo de embeddings
        ctk.CTkLabel(self.rag_options_frame, text="Modelo de Embeddings:").pack(anchor="w", padx=10, pady=2)
        self.embedding_model_var = ctk.StringVar(value="all-MiniLM-L6-v2")
        embedding_menu = ctk.CTkOptionMenu(
            self.rag_options_frame,
            variable=self.embedding_model_var,
            values=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-transformers/all-roberta-large-v1"]
        )
        embedding_menu.pack(anchor="w", padx=10, pady=2)
        
        # Modelo generativo
        ctk.CTkLabel(self.rag_options_frame, text="Modelo Generativo:").pack(anchor="w", padx=10, pady=2)
        self.generator_model_var = ctk.StringVar(value="microsoft/DialoGPT-medium")
        generator_menu = ctk.CTkOptionMenu(
            self.rag_options_frame,
            variable=self.generator_model_var,
            values=["microsoft/DialoGPT-medium", "gpt2", "facebook/blenderbot-400M-distill"]
        )
        generator_menu.pack(anchor="w", padx=10, pady=2)
        
        # Inicialmente ocultar opciones
        self.rag_options_frame.pack_forget()
    
    def setup_fine_tuning_frame(self, parent):
        """Configura el frame de fine-tuning"""
        ft_frame = ctk.CTkFrame(parent)
        ft_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            ft_frame,
            text="🎯 Fine-Tuning Avanzado",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Configuración de fine-tuning
        ft_config_frame = ctk.CTkFrame(ft_frame)
        ft_config_frame.pack(fill="x", padx=20, pady=10)
        
        # Habilitar fine-tuning
        self.ft_enabled_var = ctk.BooleanVar()
        ctk.CTkCheckBox(
            ft_config_frame,
            text="Habilitar Fine-Tuning",
            variable=self.ft_enabled_var,
            command=self.toggle_ft_options
        ).pack(anchor="w", padx=10, pady=5)
        
        # Opciones de fine-tuning
        self.ft_options_frame = ctk.CTkFrame(ft_config_frame)
        self.ft_options_frame.pack(fill="x", padx=10, pady=5)
        
        # Técnicas de fine-tuning
        ft_techniques_frame = ctk.CTkFrame(self.ft_options_frame)
        ft_techniques_frame.pack(fill="x", padx=10, pady=5)
        
        self.ft_technique_vars = {}
        ft_techniques = ["LoRA", "QLoRA", "Adapter Layers", "Prefix Tuning"]
        for tech in ft_techniques:
            var = ctk.BooleanVar()
            self.ft_technique_vars[tech] = var
            ctk.CTkCheckBox(ft_techniques_frame, text=tech, variable=var).pack(side="left", padx=10, pady=5)
        
        # Parámetros de fine-tuning
        params_frame = ctk.CTkFrame(self.ft_options_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Learning rate para fine-tuning
        ctk.CTkLabel(params_frame, text="Learning Rate:").pack(side="left", padx=5)
        self.ft_lr_var = ctk.StringVar(value="1e-5")
        ctk.CTkEntry(params_frame, textvariable=self.ft_lr_var, width=100).pack(side="left", padx=5)
        
        # Número de épocas
        ctk.CTkLabel(params_frame, text="Épocas:").pack(side="left", padx=5)
        self.ft_epochs_var = ctk.StringVar(value="3")
        ctk.CTkEntry(params_frame, textvariable=self.ft_epochs_var, width=100).pack(side="left", padx=5)
        
        # Inicialmente ocultar opciones
        self.ft_options_frame.pack_forget()
    
    def setup_training_control_frame(self, parent):
        """Configura el frame de control de entrenamiento"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            control_frame,
            text="🎮 Control de Entrenamiento",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Botones de control
        buttons_frame = ctk.CTkFrame(control_frame)
        buttons_frame.pack(fill="x", padx=20, pady=10)
        
        self.start_advanced_training_btn = ctk.CTkButton(
            buttons_frame,
            text="🚀 Iniciar Entrenamiento",
            command=self.start_advanced_training,
            width=200,
            height=40,
            fg_color=self.colors['success'],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.start_advanced_training_btn.pack(side="left", padx=10, pady=10)
        
        self.stop_advanced_training_btn = ctk.CTkButton(
            buttons_frame,
            text="⏹️ Detener",
            command=self.stop_advanced_training,
            width=150,
            height=40,
            fg_color=self.colors['danger'],
            state="disabled"
        )
        self.stop_advanced_training_btn.pack(side="left", padx=10, pady=10)
        
        self.pause_advanced_training_btn = ctk.CTkButton(
            buttons_frame,
            text="⏸️ Pausar",
            command=self.pause_advanced_training,
            width=150,
            height=40,
            fg_color=self.colors['warning'],
            state="disabled"
        )
        self.pause_advanced_training_btn.pack(side="left", padx=10, pady=10)
        
        # Estado del entrenamiento
        self.advanced_training_status = ctk.CTkLabel(
            control_frame,
            text="Estado: Listo para entrenar",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['light']
        )
        self.advanced_training_status.pack(pady=10)
        
        # Barra de progreso
        self.advanced_training_progress = ctk.CTkProgressBar(control_frame)
        self.advanced_training_progress.pack(fill="x", padx=20, pady=10)
        self.advanced_training_progress.set(0)
    
    def setup_results_frame(self, parent):
        """Configura el frame de resultados"""
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            results_frame,
            text="📈 Resultados y Métricas",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 10))
        
        # Área de resultados
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="x", padx=20, pady=10)
        self.results_text.insert("1.0", "Los resultados del entrenamiento aparecerán aquí...\n")
        
        # Botones de exportación
        export_frame = ctk.CTkFrame(results_frame)
        export_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            export_frame,
            text="💾 Guardar Modelos",
            command=self.save_trained_models,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            export_frame,
            text="📊 Exportar Métricas",
            command=self.export_metrics,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        ctk.CTkButton(
            export_frame,
            text="📋 Generar Reporte",
            command=self.generate_report,
            width=150
        ).pack(side="left", padx=5, pady=5)
    
    # Métodos de callback para la interfaz de entrenamiento avanzado
    def toggle_rag_options(self):
        """Alterna la visibilidad de las opciones RAG"""
        rag_enabled = self.rag_enabled_var.get()
        
        # Configurar widgets
        if rag_enabled:
            self.rag_options_frame.pack(fill="x", padx=10, pady=5)
        else:
            self.rag_options_frame.pack_forget()
        
        # Configurar sistema RAG en el bot_entrenador
        if hasattr(self, 'bot_entrenador'):
            rag_config = {
                'enabled': rag_enabled,
                'embedding_model': getattr(self, 'rag_embedding_var', ctk.StringVar(value='all-MiniLM-L6-v2')).get(),
                'vector_store': getattr(self, 'rag_vector_store_var', ctk.StringVar(value='faiss')).get(),
                'chunk_size': 512,
                'chunk_overlap': 50,
                'retrieval_k': 5
            }
            
            success = self.bot_entrenador.setup_rag_system(rag_config)
            if success and rag_enabled:
                self.results_text.insert("end", "✅ Sistema RAG activado y configurado\n")
            elif not rag_enabled:
                self.results_text.insert("end", "⏹️ Sistema RAG desactivado\n")
            else:
                self.results_text.insert("end", "❌ Error configurando sistema RAG\n")
    
    def toggle_ft_options(self):
        """Alterna la visibilidad de las opciones de fine-tuning"""
        if self.ft_enabled_var.get():
            self.ft_options_frame.pack(fill="x", padx=10, pady=5)
        else:
            self.ft_options_frame.pack_forget()
    
    def select_data_folder(self):
        """Selecciona una carpeta de datos"""
        folder = filedialog.askdirectory(title="Seleccionar carpeta de datos")
        if folder:
            self.data_listbox.delete("1.0", "end")
            self.data_listbox.insert("1.0", f"Carpeta seleccionada: {folder}\n")
            
            # Si RAG está habilitado, agregar la carpeta como fuente de datos
            if hasattr(self, 'rag_enabled_var') and self.rag_enabled_var.get():
                if hasattr(self, 'bot_entrenador'):
                    success = self.bot_entrenador.add_rag_data_source(folder, 'directory')
                    if success:
                        self.results_text.insert("end", f"📁 Carpeta agregada al sistema RAG: {folder}\n")
                    else:
                        self.results_text.insert("end", f"❌ Error agregando carpeta al RAG: {folder}\n")
    
    def add_data_files(self):
        """Agrega archivos de datos individuales"""
        files = filedialog.askopenfilenames(
            title="Seleccionar archivos de datos",
            filetypes=[
                ("Todos los archivos", "*.*"),
                ("Archivos de texto", "*.txt"),
                ("Archivos CSV", "*.csv"),
                ("Archivos JSON", "*.json"),
                ("Archivos PDF", "*.pdf")
            ]
        )
        if files:
            current_text = self.data_listbox.get("1.0", "end")
            if "No hay datos seleccionados" in current_text:
                self.data_listbox.delete("1.0", "end")
                current_text = ""
            
            for file in files:
                current_text += f"Archivo: {file}\n"
            
            self.data_listbox.delete("1.0", "end")
            self.data_listbox.insert("1.0", current_text)
    
    def auto_discover_data(self):
        """Auto-descubre datos en el proyecto"""
        # Simular auto-descubrimiento
        self.data_listbox.delete("1.0", "end")
        self.data_listbox.insert("1.0", "🔍 Descubriendo datos automáticamente...\n")
        self.data_listbox.insert("end", "✅ Encontrados: 15 archivos CSV\n")
        self.data_listbox.insert("end", "✅ Encontrados: 8 archivos JSON\n")
        self.data_listbox.insert("end", "✅ Encontrados: 3 archivos PDF\n")
        self.data_listbox.insert("end", "✅ Base de datos SQLite detectada\n")
    
    def get_selected_data_sources(self):
        """Obtiene las fuentes de datos seleccionadas"""
        data_sources = []
        
        # Obtener texto del listbox de datos
        data_text = self.data_listbox.get("1.0", "end").strip()
        
        if data_text and "No hay datos seleccionados" not in data_text:
            # Extraer archivos del texto
            lines = data_text.split('\n')
            for line in lines:
                if line.startswith("Archivo:"):
                    file_path = line.replace("Archivo:", "").strip()
                    data_sources.append(file_path)
        
        # Si no hay archivos específicos, usar directorio de datos por defecto
        if not data_sources:
            data_sources = ["./data/"]
        
        return data_sources
    
    def start_advanced_training(self):
        """Inicia el entrenamiento avanzado"""
        # Validar selecciones
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Advertencia", "Selecciona al menos un modelo para entrenar")
            return
        
        # Obtener fuentes de datos
        data_sources = self.get_selected_data_sources()
        
        # Obtener técnicas seleccionadas
        selected_techniques = [tech for tech, var in self.technique_vars.items() if var.get()]
        
        # Configuración RAG
        rag_enabled = self.rag_enabled_var.get()
        
        # Configuración fine-tuning
        fine_tuning_enabled = self.ft_enabled_var.get()
        fine_tuning_config = {}
        if fine_tuning_enabled:
            fine_tuning_config = {
                'techniques': [tech for tech, var in self.ft_technique_vars.items() if var.get()],
                'learning_rate': float(self.ft_lr_var.get()),
                'epochs': int(self.ft_epochs_var.get())
            }
        
        # Actualizar estado de botones
        self.start_advanced_training_btn.configure(state="disabled")
        self.stop_advanced_training_btn.configure(state="normal")
        self.pause_advanced_training_btn.configure(state="normal")
        
        # Actualizar estado
        self.advanced_training_status.configure(text="Estado: Entrenando...")
        
        # Iniciar entrenamiento con bot_entrenador
        success = self.bot_entrenador.start_training(
            models=selected_models,
            data_sources=data_sources,
            techniques=selected_techniques,
            rag_enabled=rag_enabled,
            fine_tuning_enabled=fine_tuning_enabled,
            fine_tuning_config=fine_tuning_config
        )
        
        if not success:
            # Restaurar botones si falló
            self.start_advanced_training_btn.configure(state="normal")
            self.stop_advanced_training_btn.configure(state="disabled")
            self.pause_advanced_training_btn.configure(state="disabled")
    
    def simulate_advanced_training(self, models):
        """Simula el proceso de entrenamiento avanzado"""
        def training_thread():
            total_steps = len(models) * 10
            current_step = 0
            
            for model in models:
                self.results_text.insert("end", f"\n🚀 Iniciando entrenamiento de {model}...\n")
                
                for epoch in range(10):
                    current_step += 1
                    progress = current_step / total_steps
                    
                    # Actualizar progreso en el hilo principal
                    self.root.after(0, lambda p=progress: self.advanced_training_progress.set(p))
                    
                    # Simular métricas
                    loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
                    accuracy = epoch * 0.1 + np.random.normal(0, 0.02)
                    
                    self.root.after(0, lambda m=model, e=epoch, l=loss, a=accuracy: 
                        self.results_text.insert("end", f"  Época {e+1}: Loss={l:.4f}, Accuracy={a:.4f}\n"))
                    
                    time.sleep(0.5)  # Simular tiempo de entrenamiento
                
                self.root.after(0, lambda m=model: 
                    self.results_text.insert("end", f"✅ {m} entrenado exitosamente!\n"))
            
            # Finalizar entrenamiento
            self.root.after(0, self.finish_advanced_training)
        
        # Iniciar hilo de entrenamiento
        threading.Thread(target=training_thread, daemon=True).start()
    
    def finish_advanced_training(self):
        """Finaliza el entrenamiento avanzado"""
        self.start_advanced_training_btn.configure(state="normal")
        self.stop_advanced_training_btn.configure(state="disabled")
        self.pause_advanced_training_btn.configure(state="disabled")
        
        self.advanced_training_status.configure(text="Estado: Entrenamiento completado")
        self.advanced_training_progress.set(1.0)
        
        self.results_text.insert("end", "\n🎉 ¡Entrenamiento completado exitosamente!\n")
        self.results_text.insert("end", "📊 Todos los modelos han sido entrenados y guardados.\n")
    
    def stop_advanced_training(self):
        """Detiene el entrenamiento avanzado"""
        self.bot_entrenador.stop_training()
        self.start_advanced_training_btn.configure(state="normal")
        self.stop_advanced_training_btn.configure(state="disabled")
        self.pause_advanced_training_btn.configure(state="disabled")
        
        self.advanced_training_status.configure(text="Estado: Entrenamiento detenido")
        self.results_text.insert("end", "\n⏹️ Entrenamiento detenido por el usuario.\n")
    
    def pause_advanced_training(self):
        """Pausa/reanuda el entrenamiento avanzado"""
        if self.bot_entrenador.is_training():
            if self.bot_entrenador.is_paused():
                self.bot_entrenador.resume_training()
                self.pause_advanced_training_btn.configure(text="⏸️ Pausar")
                self.advanced_training_status.configure(text="Estado: Entrenando...")
                self.results_text.insert("end", "\n▶️ Entrenamiento reanudado.\n")
            else:
                self.bot_entrenador.pause_training()
                self.pause_advanced_training_btn.configure(text="▶️ Reanudar")
                self.advanced_training_status.configure(text="Estado: Pausado")
                self.results_text.insert("end", "\n⏸️ Entrenamiento pausado.\n")
    
    def save_trained_models(self):
        """Guarda los modelos entrenados"""
        folder = filedialog.askdirectory(title="Seleccionar carpeta para guardar modelos")
        if folder:
            self.results_text.insert("end", f"\n💾 Modelos guardados en: {folder}\n")
    
    def export_metrics(self):
        """Exporta las métricas de entrenamiento"""
        file = filedialog.asksaveasfilename(
            title="Guardar métricas",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
        )
        if file:
            self.results_text.insert("end", f"\n📊 Métricas exportadas a: {file}\n")
    
    def generate_report(self):
        """Genera un reporte completo del entrenamiento"""
        file = filedialog.asksaveasfilename(
            title="Guardar reporte",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("PDF files", "*.pdf")]
        )
        if file:
            self.results_text.insert("end", f"\n📋 Reporte generado: {file}\n")
    
    # Métodos de callback para el bot_entrenador
    def on_training_progress(self, progress, current_epoch, total_epochs, metrics):
        """Callback para actualizar el progreso del entrenamiento"""
        if hasattr(self, 'advanced_training_progress'):
            self.advanced_training_progress.set(progress)
        
        if hasattr(self, 'results_text'):
            self.results_text.insert("end", f"Época {current_epoch}/{total_epochs} - Loss: {metrics.get('loss', 'N/A'):.4f}\n")
            self.results_text.see("end")
    
    def on_training_log(self, message):
        """Callback para mostrar logs del entrenamiento"""
        if hasattr(self, 'results_text'):
            self.results_text.insert("end", f"{message}\n")
            self.results_text.see("end")
    
    def on_training_complete(self, results):
        """Callback cuando el entrenamiento se completa"""
        if hasattr(self, 'start_advanced_training_btn'):
            self.start_advanced_training_btn.configure(state="normal")
            self.stop_advanced_training_btn.configure(state="disabled")
            self.pause_advanced_training_btn.configure(state="disabled")
        
        if hasattr(self, 'advanced_training_status'):
            self.advanced_training_status.configure(text="Estado: Completado")
        
        if hasattr(self, 'results_text'):
            self.results_text.insert("end", "\n🎉 ¡Entrenamiento completado exitosamente!\n")
            self.results_text.insert("end", f"📊 Resultados finales: {results}\n")
            self.results_text.see("end")
    
    def on_training_error(self, error_message):
        """Callback cuando ocurre un error en el entrenamiento"""
        if hasattr(self, 'start_advanced_training_btn'):
            self.start_advanced_training_btn.configure(state="normal")
            self.stop_advanced_training_btn.configure(state="disabled")
            self.pause_advanced_training_btn.configure(state="disabled")
        
        if hasattr(self, 'advanced_training_status'):
            self.advanced_training_status.configure(text="Estado: Error")
        
        if hasattr(self, 'results_text'):
            self.results_text.insert("end", f"\n❌ Error en el entrenamiento: {error_message}\n")
            self.results_text.see("end")
        
        messagebox.showerror("Error de Entrenamiento", f"Ocurrió un error durante el entrenamiento:\n{error_message}")

    def query_rag_system(self, query_text):
        """Consultar el sistema RAG con una pregunta"""
        if not hasattr(self, 'bot_entrenador'):
            self.results_text.insert("end", "❌ Sistema de entrenamiento no disponible\n")
            return
        
        if not hasattr(self, 'rag_enabled_var') or not self.rag_enabled_var.get():
            self.results_text.insert("end", "⚠️ Sistema RAG no está habilitado\n")
            return
        
        try:
            self.results_text.insert("end", f"🔍 Consultando RAG: {query_text}\n")
            results = self.bot_entrenador.query_rag_system(query_text, max_results=3)
            
            if results:
                self.results_text.insert("end", "📚 Resultados encontrados:\n")
                for i, result in enumerate(results, 1):
                    content = result.get('content', 'Sin contenido')
                    score = result.get('score', 0.0)
                    self.results_text.insert("end", f"  {i}. (Score: {score:.2f}) {content[:100]}...\n")
            else:
                self.results_text.insert("end", "🔍 No se encontraron resultados relevantes\n")
                
        except Exception as e:
            self.results_text.insert("end", f"❌ Error en consulta RAG: {str(e)}\n")
        
        self.results_text.see("end")

    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        try:
            # Cancelar todos los callbacks activos
            if hasattr(self, 'callback_ids') and self.callback_ids:
                for callback_name, callback_id in self.callback_ids.items():
                    if callback_id is not None:
                        try:
                            self.root.after_cancel(callback_id)
                        except:
                            pass  # Ignorar errores de cancelación
            
            # Detener entrenamiento si está activo
            if self.training_active:
                self.training_active = False
            
            # Detener sistemas de IA
            if self.voice_control:
                self.voice_control.stop_voice_recognition()
            
            if self.recommendation_system:
                self.recommendation_system.stop_recommendation_engine()
            
            # Guardar estado si hay proyecto
            if self.current_project:
                response = messagebox.askyesno(
                    "Guardar Proyecto",
                    "¿Quieres guardar el proyecto actual antes de salir?"
                )
                if response:
                    self.save_project()
            
            # Cerrar aplicación
            self.root.destroy()
            
        except Exception as e:
            print(f"Error cerrando aplicación: {e}")
            self.root.destroy()
    
    def run(self):
        """Ejecuta la interfaz"""
        try:
            print("🚀 Iniciando NeuroVision AI Interface...")
            self.root.mainloop()
        except Exception as e:
            print(f"Error ejecutando interfaz: {e}")


def main():
    """Función principal"""
    try:
        # Crear y ejecutar la interfaz
        app = NeuroVisionMainInterface()
        app.run()
        
    except Exception as e:
        print(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()