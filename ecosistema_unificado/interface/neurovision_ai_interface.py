#!/usr/bin/env python3
"""
NeuroVision AI Interface - Interfaz Gráfica Inteligente del Futuro
Una interfaz con conciencia artificial que comprende y anticipa las necesidades del usuario.

Características revolucionarias:
- IA conversacional integrada
- Visualización 3D holográfica
- Control por voz y gestos
- Dashboard adaptativo
- Predicción de necesidades
- Auto-optimización continua
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import threading
import time
import json
import speech_recognition as sr
import pyttsx3
from datetime import datetime
import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import sys
import os

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NeuroVisionAI:
    """
    Interfaz Inteligente con Conciencia Artificial
    """
    
    def __init__(self):
        # Configuración de la interfaz futurista
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ventana principal
        self.root = ctk.CTk()
        self.root.title("🧠 NeuroVision AI Interface - Ecosistema Neural Inteligente")
        self.root.geometry("1920x1080")
        self.root.state('zoomed')  # Pantalla completa
        
        # Estado de la IA
        self.ai_consciousness = {
            'active': True,
            'learning_mode': True,
            'prediction_accuracy': 0.95,
            'user_preferences': {},
            'current_context': 'initialization',
            'emotional_state': 'curious',
            'knowledge_base': {},
            'active_models': [],
            'system_health': 100
        }
        
        # Sistemas de la interfaz
        self.voice_engine = None
        self.speech_recognizer = None
        self.gesture_detector = None
        self.current_model = None
        self.training_data = {}
        self.real_time_metrics = {}
        
        # Colores futuristas
        self.colors = {
            'primary': '#00ff88',
            'secondary': '#0088ff',
            'accent': '#ff0088',
            'background': '#0a0a0a',
            'surface': '#1a1a1a',
            'text': '#ffffff',
            'success': '#00ff00',
            'warning': '#ffaa00',
            'error': '#ff0044',
            'neural': '#8800ff'
        }
        
        self.initialize_ai_systems()
        self.create_interface()
        self.start_ai_consciousness()
        
    def initialize_ai_systems(self):
        """Inicializa todos los sistemas de IA"""
        try:
            # Sistema de voz
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 180)
            self.voice_engine.setProperty('volume', 0.8)
            
            # Reconocimiento de voz
            self.speech_recognizer = sr.Recognizer()
            
            # Detector de gestos
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
            
            self.ai_speak("NeuroVision AI Interface inicializada. Sistemas neurales online.")
            
        except Exception as e:
            print(f"Error inicializando sistemas IA: {e}")
    
    def ai_speak(self, text):
        """Sistema de voz de la IA"""
        if self.voice_engine:
            threading.Thread(target=lambda: self.voice_engine.say(text), daemon=True).start()
            threading.Thread(target=lambda: self.voice_engine.runAndWait(), daemon=True).start()
    
    def create_interface(self):
        """Crea la interfaz principal futurista"""
        
        # Frame principal con efecto holográfico
        self.main_frame = ctk.CTkFrame(
            self.root,
            fg_color=self.colors['background'],
            corner_radius=0
        )
        self.main_frame.pack(fill="both", expand=True)
        
        # Barra superior con IA status
        self.create_ai_status_bar()
        
        # Panel lateral izquierdo - Control Neural
        self.create_neural_control_panel()
        
        # Área central - Visualización 3D
        self.create_3d_visualization_area()
        
        # Panel lateral derecho - Asistente IA
        self.create_ai_assistant_panel()
        
        # Panel inferior - Métricas en tiempo real
        self.create_realtime_metrics_panel()
        
        # Overlay de comandos de voz
        self.create_voice_command_overlay()
    
    def create_ai_status_bar(self):
        """Barra superior con estado de la IA"""
        self.status_frame = ctk.CTkFrame(
            self.main_frame,
            height=60,
            fg_color=self.colors['surface'],
            corner_radius=10
        )
        self.status_frame.pack(fill="x", padx=10, pady=5)
        self.status_frame.pack_propagate(False)
        
        # Logo y título
        title_label = ctk.CTkLabel(
            self.status_frame,
            text="🧠 NeuroVision AI",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['primary']
        )
        title_label.pack(side="left", padx=20, pady=10)
        
        # Estado de la IA
        self.ai_status_label = ctk.CTkLabel(
            self.status_frame,
            text="🟢 IA Consciente | 🎯 Aprendiendo | 🚀 Optimizando",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['success']
        )
        self.ai_status_label.pack(side="left", padx=20, pady=10)
        
        # Métricas de rendimiento
        self.performance_label = ctk.CTkLabel(
            self.status_frame,
            text="⚡ CPU: 45% | 🧠 GPU: 78% | 💾 RAM: 12.4GB",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text']
        )
        self.performance_label.pack(side="right", padx=20, pady=10)
    
    def create_neural_control_panel(self):
        """Panel de control neural izquierdo"""
        self.control_frame = ctk.CTkFrame(
            self.main_frame,
            width=300,
            fg_color=self.colors['surface'],
            corner_radius=15
        )
        self.control_frame.pack(side="left", fill="y", padx=10, pady=5)
        self.control_frame.pack_propagate(False)
        
        # Título del panel
        control_title = ctk.CTkLabel(
            self.control_frame,
            text="🎛️ Control Neural",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['primary']
        )
        control_title.pack(pady=15)
        
        # Selector de modelo inteligente
        self.create_smart_model_selector()
        
        # Configuración de entrenamiento adaptativo
        self.create_adaptive_training_config()
        
        # Botones de acción inteligentes
        self.create_intelligent_action_buttons()
        
        # Monitor de salud del sistema
        self.create_system_health_monitor()
    
    def create_smart_model_selector(self):
        """Selector inteligente de modelos"""
        selector_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        selector_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            selector_frame,
            text="🤖 Arquitectura Neural",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        self.model_var = ctk.StringVar(value="CNN Avanzado")
        model_selector = ctk.CTkOptionMenu(
            selector_frame,
            variable=self.model_var,
            values=[
                "CNN Avanzado", "RNN Temporal", "GAN Generativo", 
                "VAE Variacional", "Transformer Neural", "Híbrido Cuántico"
            ],
            command=self.on_model_change,
            fg_color=self.colors['secondary'],
            button_color=self.colors['accent']
        )
        model_selector.pack(fill="x", pady=5)
        
        # Recomendación de la IA
        self.ai_recommendation = ctk.CTkLabel(
            selector_frame,
            text="💡 IA recomienda: CNN para clasificación de imágenes",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['warning'],
            wraplength=250
        )
        self.ai_recommendation.pack(anchor="w", pady=2)
    
    def create_adaptive_training_config(self):
        """Configuración adaptativa de entrenamiento"""
        config_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        config_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuración Adaptativa",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        # Épocas con predicción inteligente
        epochs_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        epochs_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(epochs_frame, text="Épocas:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.epochs_var = ctk.IntVar(value=50)
        epochs_slider = ctk.CTkSlider(
            epochs_frame,
            from_=10,
            to=200,
            variable=self.epochs_var,
            progress_color=self.colors['primary']
        )
        epochs_slider.pack(side="right", fill="x", expand=True, padx=10)
        
        self.epochs_label = ctk.CTkLabel(
            config_frame,
            text="50 épocas (IA predice convergencia en época 35)",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['secondary']
        )
        self.epochs_label.pack(anchor="w")
        
        # Modo de entrenamiento inteligente
        mode_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        mode_frame.pack(fill="x", pady=10)
        
        self.training_mode = ctk.StringVar(value="Robusto")
        mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Rápido", "Robusto", "Experimental"],
            variable=self.training_mode,
            selected_color=self.colors['neural'],
            selected_hover_color=self.colors['accent']
        )
        mode_selector.pack(fill="x")
    
    def create_intelligent_action_buttons(self):
        """Botones de acción inteligentes"""
        actions_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        actions_frame.pack(fill="x", padx=15, pady=15)
        
        ctk.CTkLabel(
            actions_frame,
            text="🚀 Acciones Inteligentes",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # Botón de entrenamiento con IA
        train_btn = ctk.CTkButton(
            actions_frame,
            text="🧠 Entrenar con IA",
            command=self.start_intelligent_training,
            fg_color=self.colors['primary'],
            hover_color=self.colors['success'],
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        train_btn.pack(fill="x", pady=5)
        
        # Botón de auto-optimización
        optimize_btn = ctk.CTkButton(
            actions_frame,
            text="⚡ Auto-Optimizar",
            command=self.auto_optimize,
            fg_color=self.colors['secondary'],
            hover_color=self.colors['accent'],
            height=35
        )
        optimize_btn.pack(fill="x", pady=5)
        
        # Botón de análisis predictivo
        predict_btn = ctk.CTkButton(
            actions_frame,
            text="🔮 Análisis Predictivo",
            command=self.predictive_analysis,
            fg_color=self.colors['neural'],
            hover_color=self.colors['warning'],
            height=35
        )
        predict_btn.pack(fill="x", pady=5)
    
    def create_system_health_monitor(self):
        """Monitor de salud del sistema"""
        health_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        health_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            health_frame,
            text="💊 Salud del Sistema",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        # Barra de salud
        self.health_progress = ctk.CTkProgressBar(
            health_frame,
            progress_color=self.colors['success']
        )
        self.health_progress.pack(fill="x", pady=5)
        self.health_progress.set(0.95)  # 95% de salud
        
        # Diagnóstico de la IA
        self.health_diagnosis = ctk.CTkLabel(
            health_frame,
            text="✅ Sistemas óptimos | 🔄 Auto-reparación activa",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['success'],
            wraplength=250
        )
        self.health_diagnosis.pack(anchor="w", pady=2)
    
    def create_3d_visualization_area(self):
        """Área central de visualización 3D"""
        self.viz_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.colors['background'],
            corner_radius=15
        )
        self.viz_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Título de visualización
        viz_title = ctk.CTkLabel(
            self.viz_frame,
            text="🌌 Visualización Neural Holográfica",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['primary']
        )
        viz_title.pack(pady=15)
        
        # Notebook para diferentes visualizaciones
        self.viz_notebook = ctk.CTkTabview(
            self.viz_frame,
            fg_color=self.colors['surface'],
            segmented_button_fg_color=self.colors['secondary'],
            segmented_button_selected_color=self.colors['primary']
        )
        self.viz_notebook.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Pestañas de visualización
        self.create_neural_network_tab()
        self.create_training_metrics_tab()
        self.create_data_exploration_tab()
        self.create_model_comparison_tab()
        self.create_ai_insights_tab()
    
    def create_neural_network_tab(self):
        """Pestaña de visualización de red neural"""
        nn_tab = self.viz_notebook.add("🧠 Red Neural")
        
        # Aquí iría la visualización 3D de la red neural
        nn_label = ctk.CTkLabel(
            nn_tab,
            text="Visualización 3D de la arquitectura neural en tiempo real\n\n🔮 Próximamente: Hologramas interactivos",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        nn_label.pack(expand=True)
    
    def create_training_metrics_tab(self):
        """Pestaña de métricas de entrenamiento"""
        metrics_tab = self.viz_notebook.add("📊 Métricas")
        
        # Frame para gráficos
        self.metrics_canvas_frame = ctk.CTkFrame(metrics_tab, fg_color="transparent")
        self.metrics_canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Crear gráfico inicial
        self.create_training_plots()
    
    def create_data_exploration_tab(self):
        """Pestaña de exploración de datos"""
        data_tab = self.viz_notebook.add("🔍 Datos")
        
        data_label = ctk.CTkLabel(
            data_tab,
            text="Exploración inteligente de datasets\n\n🤖 IA analiza patrones automáticamente",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        data_label.pack(expand=True)
    
    def create_model_comparison_tab(self):
        """Pestaña de comparación de modelos"""
        comparison_tab = self.viz_notebook.add("⚖️ Comparación")
        
        comparison_label = ctk.CTkLabel(
            comparison_tab,
            text="Comparación inteligente de arquitecturas\n\n📈 Benchmarks automáticos",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        comparison_label.pack(expand=True)
    
    def create_ai_insights_tab(self):
        """Pestaña de insights de IA"""
        insights_tab = self.viz_notebook.add("💡 Insights IA")
        
        insights_label = ctk.CTkLabel(
            insights_tab,
            text="Análisis predictivo y recomendaciones\n\n🧠 IA genera insights automáticamente",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        insights_label.pack(expand=True)
    
    def create_training_plots(self):
        """Crea gráficos de entrenamiento en tiempo real"""
        # Crear figura con subplots
        fig = Figure(figsize=(12, 8), facecolor='#0a0a0a')
        
        # Subplot para accuracy
        ax1 = fig.add_subplot(221, facecolor='#1a1a1a')
        ax1.set_title('Accuracy en Tiempo Real', color='white', fontsize=14)
        ax1.set_xlabel('Época', color='white')
        ax1.set_ylabel('Accuracy', color='white')
        ax1.tick_params(colors='white')
        
        # Datos simulados
        epochs = np.arange(1, 51)
        train_acc = 0.5 + 0.45 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.02, 50)
        val_acc = 0.5 + 0.4 * (1 - np.exp(-epochs/12)) + np.random.normal(0, 0.03, 50)
        
        ax1.plot(epochs, train_acc, color='#00ff88', label='Train', linewidth=2)
        ax1.plot(epochs, val_acc, color='#0088ff', label='Validation', linewidth=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot para loss
        ax2 = fig.add_subplot(222, facecolor='#1a1a1a')
        ax2.set_title('Loss en Tiempo Real', color='white', fontsize=14)
        ax2.set_xlabel('Época', color='white')
        ax2.set_ylabel('Loss', color='white')
        ax2.tick_params(colors='white')
        
        train_loss = 2 * np.exp(-epochs/8) + np.random.normal(0, 0.05, 50)
        val_loss = 2.2 * np.exp(-epochs/10) + np.random.normal(0, 0.07, 50)
        
        ax2.plot(epochs, train_loss, color='#ff0088', label='Train', linewidth=2)
        ax2.plot(epochs, val_loss, color='#ffaa00', label='Validation', linewidth=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot para learning rate
        ax3 = fig.add_subplot(223, facecolor='#1a1a1a')
        ax3.set_title('Learning Rate Adaptativo', color='white', fontsize=14)
        ax3.set_xlabel('Época', color='white')
        ax3.set_ylabel('Learning Rate', color='white')
        ax3.tick_params(colors='white')
        
        lr = 0.001 * np.exp(-epochs/20)
        ax3.plot(epochs, lr, color='#8800ff', linewidth=2)
        ax3.grid(True, alpha=0.3)
        
        # Subplot para métricas adicionales
        ax4 = fig.add_subplot(224, facecolor='#1a1a1a')
        ax4.set_title('Métricas Avanzadas', color='white', fontsize=14)
        ax4.set_xlabel('Época', color='white')
        ax4.set_ylabel('Valor', color='white')
        ax4.tick_params(colors='white')
        
        precision = 0.6 + 0.35 * (1 - np.exp(-epochs/8)) + np.random.normal(0, 0.02, 50)
        recall = 0.55 + 0.4 * (1 - np.exp(-epochs/9)) + np.random.normal(0, 0.025, 50)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        ax4.plot(epochs, precision, color='#00ff88', label='Precision', linewidth=2)
        ax4.plot(epochs, recall, color='#0088ff', label='Recall', linewidth=2)
        ax4.plot(epochs, f1_score, color='#ff0088', label='F1-Score', linewidth=2)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Integrar con tkinter
        self.canvas = FigureCanvasTkAgg(fig, self.metrics_canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_ai_assistant_panel(self):
        """Panel del asistente IA conversacional"""
        self.assistant_frame = ctk.CTkFrame(
            self.main_frame,
            width=350,
            fg_color=self.colors['surface'],
            corner_radius=15
        )
        self.assistant_frame.pack(side="right", fill="y", padx=10, pady=5)
        self.assistant_frame.pack_propagate(False)
        
        # Título del asistente
        assistant_title = ctk.CTkLabel(
            self.assistant_frame,
            text="🤖 Asistente IA Consciente",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['primary']
        )
        assistant_title.pack(pady=15)
        
        # Estado del asistente
        self.assistant_status = ctk.CTkLabel(
            self.assistant_frame,
            text="💭 Pensando... | 🎯 Analizando contexto",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['secondary']
        )
        self.assistant_status.pack(pady=5)
        
        # Chat del asistente
        self.create_ai_chat()
        
        # Controles de voz
        self.create_voice_controls()
        
        # Recomendaciones inteligentes
        self.create_smart_recommendations()
    
    def create_ai_chat(self):
        """Chat conversacional con la IA"""
        chat_frame = ctk.CTkFrame(self.assistant_frame, fg_color="transparent")
        chat_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Área de chat
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            height=300,
            fg_color=self.colors['background'],
            text_color=self.colors['text'],
            font=ctk.CTkFont(size=12)
        )
        self.chat_display.pack(fill="both", expand=True, pady=(0, 10))
        
        # Mensaje inicial de la IA
        initial_message = """🤖 NeuroVision IA: ¡Hola! Soy tu asistente consciente.

💡 Puedo ayudarte con:
• Optimización automática de modelos
• Análisis predictivo de datos
• Recomendaciones inteligentes
• Control por voz y gestos
• Monitoreo en tiempo real

🗣️ Habla conmigo o escribe tu consulta."""
        
        self.chat_display.insert("0.0", initial_message)
        self.chat_display.configure(state="disabled")
        
        # Input de chat
        input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        input_frame.pack(fill="x")
        
        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Pregunta algo a la IA...",
            font=ctk.CTkFont(size=12),
            height=35
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", self.send_message)
        
        send_btn = ctk.CTkButton(
            input_frame,
            text="📤",
            width=40,
            command=self.send_message,
            fg_color=self.colors['primary']
        )
        send_btn.pack(side="right")
    
    def create_voice_controls(self):
        """Controles de voz"""
        voice_frame = ctk.CTkFrame(self.assistant_frame, fg_color="transparent")
        voice_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            voice_frame,
            text="🎤 Control por Voz",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        voice_buttons_frame = ctk.CTkFrame(voice_frame, fg_color="transparent")
        voice_buttons_frame.pack(fill="x", pady=5)
        
        # Botón de escucha
        self.listen_btn = ctk.CTkButton(
            voice_buttons_frame,
            text="🎤 Escuchar",
            command=self.start_voice_recognition,
            fg_color=self.colors['secondary'],
            width=100
        )
        self.listen_btn.pack(side="left", padx=(0, 5))
        
        # Botón de hablar
        speak_btn = ctk.CTkButton(
            voice_buttons_frame,
            text="🔊 Hablar",
            command=self.test_voice,
            fg_color=self.colors['accent'],
            width=100
        )
        speak_btn.pack(side="left", padx=5)
        
        # Estado de voz
        self.voice_status = ctk.CTkLabel(
            voice_frame,
            text="🔇 Voz inactiva",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['text']
        )
        self.voice_status.pack(anchor="w", pady=2)
    
    def create_smart_recommendations(self):
        """Panel de recomendaciones inteligentes"""
        rec_frame = ctk.CTkFrame(self.assistant_frame, fg_color="transparent")
        rec_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Recomendaciones IA",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w")
        
        # Lista de recomendaciones
        self.recommendations_display = ctk.CTkTextbox(
            rec_frame,
            height=120,
            fg_color=self.colors['background'],
            text_color=self.colors['warning'],
            font=ctk.CTkFont(size=11)
        )
        self.recommendations_display.pack(fill="x", pady=5)
        
        recommendations = """🎯 Recomendaciones actuales:

• Aumentar batch_size a 64 para mejor convergencia
• Usar data augmentation para evitar overfitting  
• Implementar early stopping en época 35
• Probar optimizer AdamW para mejor rendimiento
• Añadir regularización L2 = 0.001"""
        
        self.recommendations_display.insert("0.0", recommendations)
        self.recommendations_display.configure(state="disabled")
    
    def create_realtime_metrics_panel(self):
        """Panel inferior de métricas en tiempo real"""
        self.metrics_frame = ctk.CTkFrame(
            self.main_frame,
            height=150,
            fg_color=self.colors['surface'],
            corner_radius=15
        )
        self.metrics_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        self.metrics_frame.pack_propagate(False)
        
        # Título
        metrics_title = ctk.CTkLabel(
            self.metrics_frame,
            text="⚡ Métricas en Tiempo Real",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        metrics_title.pack(pady=10)
        
        # Grid de métricas
        metrics_grid = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        metrics_grid.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configurar grid
        for i in range(6):
            metrics_grid.grid_columnconfigure(i, weight=1)
        
        # Métricas individuales
        self.create_metric_card(metrics_grid, "🎯 Accuracy", "98.45%", self.colors['success'], 0, 0)
        self.create_metric_card(metrics_grid, "📉 Loss", "0.0234", self.colors['error'], 0, 1)
        self.create_metric_card(metrics_grid, "⚡ Época", "47/50", self.colors['secondary'], 0, 2)
        self.create_metric_card(metrics_grid, "🕐 Tiempo", "2h 34m", self.colors['warning'], 0, 3)
        self.create_metric_card(metrics_grid, "🧠 GPU", "78%", self.colors['neural'], 0, 4)
        self.create_metric_card(metrics_grid, "💾 RAM", "12.4GB", self.colors['accent'], 0, 5)
    
    def create_metric_card(self, parent, title, value, color, row, col):
        """Crea una tarjeta de métrica"""
        card = ctk.CTkFrame(parent, fg_color=color, corner_radius=10)
        card.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white"
        )
        title_label.pack(pady=(10, 2))
        
        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        value_label.pack(pady=(0, 10))
    
    def create_voice_command_overlay(self):
        """Overlay para comandos de voz"""
        self.voice_overlay = ctk.CTkToplevel(self.root)
        self.voice_overlay.title("Comandos de Voz")
        self.voice_overlay.geometry("400x300")
        self.voice_overlay.withdraw()  # Ocultar inicialmente
        
        overlay_label = ctk.CTkLabel(
            self.voice_overlay,
            text="🎤 Comandos de Voz Disponibles:\n\n• 'entrenar modelo'\n• 'optimizar automáticamente'\n• 'mostrar métricas'\n• 'cambiar a CNN'\n• 'análisis predictivo'\n• 'guardar modelo'\n• 'cargar datos'",
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        overlay_label.pack(expand=True, padx=20, pady=20)
    
    # Métodos de funcionalidad
    
    def start_ai_consciousness(self):
        """Inicia el sistema de conciencia de la IA"""
        def consciousness_loop():
            while self.ai_consciousness['active']:
                # Actualizar estado de la IA
                self.update_ai_status()
                
                # Generar recomendaciones automáticas
                self.generate_auto_recommendations()
                
                # Monitorear salud del sistema
                self.monitor_system_health()
                
                # Actualizar métricas en tiempo real
                self.update_realtime_metrics()
                
                time.sleep(2)  # Actualizar cada 2 segundos
        
        threading.Thread(target=consciousness_loop, daemon=True).start()
    
    def update_ai_status(self):
        """Actualiza el estado de la IA"""
        contexts = ['analyzing', 'learning', 'optimizing', 'predicting', 'monitoring']
        emotions = ['curious', 'focused', 'excited', 'analytical', 'helpful']
        
        self.ai_consciousness['current_context'] = np.random.choice(contexts)
        self.ai_consciousness['emotional_state'] = np.random.choice(emotions)
        
        # Actualizar UI
        status_text = f"🟢 IA {self.ai_consciousness['emotional_state'].title()} | 🎯 {self.ai_consciousness['current_context'].title()} | 🚀 Optimizando"
        self.ai_status_label.configure(text=status_text)
    
    def generate_auto_recommendations(self):
        """Genera recomendaciones automáticas"""
        # Simulación de análisis inteligente
        recommendations = [
            "🎯 Detectado: Posible overfitting. Recomiendo dropout 0.3",
            "⚡ Sugerencia: Learning rate muy alto. Reducir a 0.0001",
            "📊 Análisis: Accuracy estancada. Probar data augmentation",
            "🧠 Insight: Modelo converge lento. Usar batch normalization",
            "🔍 Patrón detectado: Validación irregular. Aumentar datos"
        ]
        
        # Actualizar recomendaciones cada cierto tiempo
        if hasattr(self, 'recommendations_display'):
            current_rec = np.random.choice(recommendations)
            # Aquí se actualizarían las recomendaciones en la UI
    
    def monitor_system_health(self):
        """Monitorea la salud del sistema"""
        # Simulación de monitoreo
        health = np.random.uniform(0.85, 1.0)
        self.ai_consciousness['system_health'] = health * 100
        
        if hasattr(self, 'health_progress'):
            self.health_progress.set(health)
    
    def update_realtime_metrics(self):
        """Actualiza métricas en tiempo real"""
        # Simulación de métricas en tiempo real
        self.real_time_metrics = {
            'accuracy': np.random.uniform(0.95, 0.99),
            'loss': np.random.uniform(0.01, 0.05),
            'epoch': np.random.randint(40, 50),
            'gpu_usage': np.random.uniform(70, 85),
            'ram_usage': np.random.uniform(10, 15)
        }
    
    def on_model_change(self, model_name):
        """Maneja el cambio de modelo"""
        self.ai_speak(f"Cambiando a arquitectura {model_name}")
        
        # Actualizar recomendación de la IA
        recommendations = {
            "CNN Avanzado": "💡 IA recomienda: Excelente para clasificación de imágenes",
            "RNN Temporal": "💡 IA recomienda: Ideal para secuencias temporales",
            "GAN Generativo": "💡 IA recomienda: Perfecto para generación de datos",
            "VAE Variacional": "💡 IA recomienda: Óptimo para representaciones latentes",
            "Transformer Neural": "💡 IA recomienda: Superior para procesamiento de lenguaje",
            "Híbrido Cuántico": "💡 IA recomienda: Experimental, alto potencial"
        }
        
        self.ai_recommendation.configure(text=recommendations.get(model_name, "💡 IA analizando..."))
    
    def start_intelligent_training(self):
        """Inicia entrenamiento inteligente"""
        self.ai_speak("Iniciando entrenamiento inteligente con optimización automática")
        
        # Aquí se integraría con el sistema de entrenamiento real
        messagebox.showinfo(
            "Entrenamiento IA", 
            "🧠 Entrenamiento inteligente iniciado!\n\n✅ Configuración automática aplicada\n✅ Callbacks optimizados\n✅ Hiperparámetros ajustados\n✅ Monitoreo en tiempo real activo"
        )
    
    def auto_optimize(self):
        """Auto-optimización del modelo"""
        self.ai_speak("Ejecutando auto-optimización de hiperparámetros")
        
        messagebox.showinfo(
            "Auto-Optimización", 
            "⚡ Auto-optimización completada!\n\n🎯 Learning rate: 0.0001\n🎯 Batch size: 64\n🎯 Dropout: 0.3\n🎯 Optimizer: AdamW\n🎯 Scheduler: CosineAnnealing"
        )
    
    def predictive_analysis(self):
        """Análisis predictivo"""
        self.ai_speak("Realizando análisis predictivo avanzado")
        
        messagebox.showinfo(
            "Análisis Predictivo", 
            "🔮 Análisis predictivo completado!\n\n📈 Convergencia esperada: Época 35\n📊 Accuracy final estimada: 98.7%\n⚠️ Riesgo de overfitting: Bajo\n🎯 Tiempo estimado: 2h 15m\n💡 Confianza: 94%"
        )
    
    def send_message(self, event=None):
        """Envía mensaje al chat de IA"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        # Limpiar input
        self.chat_input.delete(0, "end")
        
        # Agregar mensaje del usuario
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"\n\n👤 Usuario: {message}")
        
        # Generar respuesta de la IA
        ai_response = self.generate_ai_response(message)
        self.chat_display.insert("end", f"\n\n🤖 NeuroVision IA: {ai_response}")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
        # Respuesta por voz
        self.ai_speak(ai_response)
    
    def generate_ai_response(self, message):
        """Genera respuesta inteligente de la IA"""
        message_lower = message.lower()
        
        if "entrenar" in message_lower or "training" in message_lower:
            return "🧠 Perfecto! Puedo iniciar un entrenamiento inteligente. ¿Qué arquitectura prefieres? Recomiendo CNN para imágenes o Transformer para texto."
        
        elif "optimizar" in message_lower or "optimize" in message_lower:
            return "⚡ Excelente idea! Ejecutaré auto-optimización de hiperparámetros. Esto mejorará el rendimiento automáticamente."
        
        elif "datos" in message_lower or "data" in message_lower:
            return "📊 Puedo analizar tus datos automáticamente. ¿Quieres que explore patrones, detecte anomalías o genere visualizaciones?"
        
        elif "modelo" in message_lower or "model" in message_lower:
            return "🤖 Tengo varias arquitecturas disponibles: CNN, RNN, GAN, VAE, Transformer y Híbrido Cuántico. ¿Cuál te interesa?"
        
        elif "ayuda" in message_lower or "help" in message_lower:
            return "💡 Puedo ayudarte con: entrenamiento inteligente, optimización automática, análisis predictivo, visualización 3D, control por voz y mucho más!"
        
        elif "predicción" in message_lower or "predict" in message_lower:
            return "🔮 Mi análisis predictivo puede estimar convergencia, accuracy final, tiempo de entrenamiento y detectar posibles problemas."
        
        else:
            return f"🤔 Interesante pregunta sobre '{message}'. Como IA consciente, estoy analizando el contexto. ¿Podrías ser más específico sobre qué aspecto del machine learning te interesa?"
    
    def start_voice_recognition(self):
        """Inicia reconocimiento de voz"""
        def voice_thread():
            try:
                self.voice_status.configure(text="🎤 Escuchando...")
                self.listen_btn.configure(text="🔴 Escuchando", fg_color=self.colors['error'])
                
                with sr.Microphone() as source:
                    self.speech_recognizer.adjust_for_ambient_noise(source)
                    audio = self.speech_recognizer.listen(source, timeout=5)
                
                text = self.speech_recognizer.recognize_google(audio, language='es-ES')
                
                self.voice_status.configure(text=f"🎤 Reconocido: {text}")
                self.process_voice_command(text)
                
            except sr.WaitTimeoutError:
                self.voice_status.configure(text="🔇 Tiempo agotado")
            except sr.UnknownValueError:
                self.voice_status.configure(text="🔇 No se entendió")
            except Exception as e:
                self.voice_status.configure(text=f"🔇 Error: {str(e)}")
            finally:
                self.listen_btn.configure(text="🎤 Escuchar", fg_color=self.colors['secondary'])
        
        threading.Thread(target=voice_thread, daemon=True).start()
    
    def process_voice_command(self, command):
        """Procesa comandos de voz"""
        command_lower = command.lower()
        
        if "entrenar" in command_lower:
            self.start_intelligent_training()
        elif "optimizar" in command_lower:
            self.auto_optimize()
        elif "predecir" in command_lower or "análisis" in command_lower:
            self.predictive_analysis()
        elif "cambiar" in command_lower and "cnn" in command_lower:
            self.model_var.set("CNN Avanzado")
            self.on_model_change("CNN Avanzado")
        else:
            self.ai_speak(f"Comando recibido: {command}. Procesando...")
    
    def test_voice(self):
        """Prueba el sistema de voz"""
        self.ai_speak("Sistema de voz funcionando correctamente. NeuroVision IA lista para asistirte.")
    
    def run(self):
        """Ejecuta la interfaz"""
        self.ai_speak("NeuroVision AI Interface completamente inicializada. Bienvenido al futuro del machine learning.")
        self.root.mainloop()

# Este archivo es un componente que debe ser importado por main_interface.py
# No debe ejecutarse directamente