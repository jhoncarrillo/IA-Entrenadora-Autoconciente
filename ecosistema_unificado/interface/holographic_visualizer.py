#!/usr/bin/env python3
"""
Holographic Visualizer - Sistema de Visualización 3D Holográfica
Visualización avanzada de redes neurales, datos y métricas en tiempo real
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
import threading
import time
from datetime import datetime

class HolographicVisualizer:
    """
    Sistema de visualización holográfica 3D para redes neurales
    """
    
    def __init__(self):
        self.colors = {
            'primary': '#00ff88',
            'secondary': '#0088ff', 
            'accent': '#ff0088',
            'neural': '#8800ff',
            'success': '#00ff00',
            'warning': '#ffaa00',
            'error': '#ff0044'
        }
        
        self.animation_data = {}
        self.real_time_data = {}
        self.network_topology = None
        
    def create_neural_network_3d(self, model=None, layer_info=None):
        """
        Crea visualización 3D de la arquitectura de red neural
        """
        if layer_info is None:
            # Arquitectura de ejemplo
            layer_info = [
                {'name': 'Input', 'neurons': 784, 'type': 'input'},
                {'name': 'Conv2D_1', 'neurons': 32, 'type': 'conv'},
                {'name': 'MaxPool_1', 'neurons': 32, 'type': 'pool'},
                {'name': 'Conv2D_2', 'neurons': 64, 'type': 'conv'},
                {'name': 'MaxPool_2', 'neurons': 64, 'type': 'pool'},
                {'name': 'Flatten', 'neurons': 1024, 'type': 'flatten'},
                {'name': 'Dense_1', 'neurons': 128, 'type': 'dense'},
                {'name': 'Dropout', 'neurons': 128, 'type': 'dropout'},
                {'name': 'Output', 'neurons': 10, 'type': 'output'}
            ]
        
        fig = go.Figure()
        
        # Posiciones de las capas
        layer_positions = []
        max_neurons = max([layer['neurons'] for layer in layer_info])
        
        for i, layer in enumerate(layer_info):
            z = i * 2  # Espaciado entre capas
            neurons = layer['neurons']
            
            # Distribución circular de neuronas
            if neurons <= 20:
                # Pocas neuronas: distribución circular
                angles = np.linspace(0, 2*np.pi, neurons, endpoint=False)
                radius = 1
                x = radius * np.cos(angles)
                y = radius * np.sin(angles)
            else:
                # Muchas neuronas: distribución en grid 3D
                grid_size = int(np.ceil(np.sqrt(neurons)))
                x, y = np.meshgrid(
                    np.linspace(-1, 1, grid_size),
                    np.linspace(-1, 1, grid_size)
                )
                x = x.flatten()[:neurons]
                y = y.flatten()[:neurons]
            
            z_coords = np.full(len(x), z)
            
            # Color según tipo de capa
            color_map = {
                'input': self.colors['primary'],
                'conv': self.colors['secondary'],
                'pool': self.colors['accent'],
                'dense': self.colors['neural'],
                'dropout': self.colors['warning'],
                'flatten': self.colors['success'],
                'output': self.colors['error']
            }
            
            color = color_map.get(layer['type'], '#ffffff')
            
            # Agregar neuronas
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                name=f"{layer['name']} ({neurons})",
                text=[f"Neurona {j+1}<br>Capa: {layer['name']}" for j in range(len(x))],
                hovertemplate="<b>%{text}</b><br>Posición: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ))
            
            layer_positions.append({
                'layer': layer,
                'x': x,
                'y': y,
                'z': z_coords,
                'color': color
            })
        
        # Agregar conexiones entre capas
        self.add_layer_connections(fig, layer_positions)
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "🧠 Arquitectura Neural 3D Holográfica",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(
                xaxis=dict(
                    title="X",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.1)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
                yaxis=dict(
                    title="Y", 
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.1)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
                zaxis=dict(
                    title="Capas",
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.1)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.2)"
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor="rgba(0,0,0,0.9)"
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                borderwidth=1
            )
        )
        
        return fig
    
    def create_neural_network_viz(self, parent_frame):
        """
        Crea widget de visualización de red neural para la interfaz
        """
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
        import matplotlib.pyplot as plt
        
        # Crear frame para la visualización
        viz_frame = tk.Frame(parent_frame, bg='#1a1a1a')
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear figura de matplotlib
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Crear visualización simple de red neural
        layers = [4, 6, 4, 2]  # Ejemplo de arquitectura
        layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
        
        # Posiciones de las neuronas
        max_neurons = max(layers)
        for i, (layer_size, name) in enumerate(zip(layers, layer_names)):
            x = i
            y_positions = np.linspace(-max_neurons/2, max_neurons/2, layer_size)
            
            # Dibujar neuronas
            for y in y_positions:
                circle = plt.Circle((x, y), 0.3, color=self.colors['primary'], alpha=0.7)
                ax.add_patch(circle)
            
            # Etiquetas de capas
            ax.text(x, max_neurons/2 + 1, name, ha='center', va='bottom', 
                   color='white', fontsize=10, weight='bold')
        
        # Dibujar conexiones entre capas
        for i in range(len(layers)-1):
            for j in range(layers[i]):
                for k in range(layers[i+1]):
                    y1 = np.linspace(-max_neurons/2, max_neurons/2, layers[i])[j]
                    y2 = np.linspace(-max_neurons/2, max_neurons/2, layers[i+1])[k]
                    ax.plot([i, i+1], [y1, y2], color=self.colors['secondary'], 
                           alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-0.5, len(layers)-0.5)
        ax.set_ylim(-max_neurons/2-1, max_neurons/2+2)
        ax.set_title('🧠 Arquitectura de Red Neural', color='white', fontsize=14, weight='bold')
        ax.axis('off')
        
        # Integrar con tkinter
        canvas = FigureCanvasTkinter(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return viz_frame
    
    def create_data_exploration_viz(self, parent_frame):
        """
        Crea widget de visualización de exploración de datos
        """
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
        import matplotlib.pyplot as plt
        
        # Crear frame para la visualización
        viz_frame = tk.Frame(parent_frame, bg='#1a1a1a')
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), facecolor='#1a1a1a')
        
        # Datos de ejemplo
        np.random.seed(42)
        data = np.random.randn(1000, 2)
        
        # Gráfico 1: Scatter plot
        ax1.scatter(data[:, 0], data[:, 1], c=self.colors['primary'], alpha=0.6, s=20)
        ax1.set_title('📊 Distribución de Datos', color='white', fontsize=12)
        ax1.set_facecolor('#1a1a1a')
        ax1.tick_params(colors='white')
        
        # Gráfico 2: Histograma
        ax2.hist(data[:, 0], bins=30, color=self.colors['secondary'], alpha=0.7)
        ax2.set_title('📈 Distribución Feature 1', color='white', fontsize=12)
        ax2.set_facecolor('#1a1a1a')
        ax2.tick_params(colors='white')
        
        # Gráfico 3: Box plot
        ax3.boxplot([data[:, 0], data[:, 1]], patch_artist=True,
                   boxprops=dict(facecolor=self.colors['accent'], alpha=0.7))
        ax3.set_title('📦 Box Plot Features', color='white', fontsize=12)
        ax3.set_facecolor('#1a1a1a')
        ax3.tick_params(colors='white')
        
        # Gráfico 4: Correlación
        correlation = np.corrcoef(data.T)
        im = ax4.imshow(correlation, cmap='viridis', aspect='auto')
        ax4.set_title('🔗 Matriz de Correlación', color='white', fontsize=12)
        ax4.set_facecolor('#1a1a1a')
        ax4.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Integrar con tkinter
        canvas = FigureCanvasTkinter(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return viz_frame
    
    def create_attention_heatmap_viz(self, parent_frame):
        """
        Crea widget de visualización de mapa de atención
        """
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
        import matplotlib.pyplot as plt
        
        # Crear frame para la visualización
        viz_frame = tk.Frame(parent_frame, bg='#1a1a1a')
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Generar datos de atención de ejemplo
        seq_length = 20
        attention_matrix = np.random.rand(seq_length, seq_length)
        
        # Hacer la matriz más realista (más atención a posiciones cercanas)
        for i in range(seq_length):
            for j in range(seq_length):
                distance = abs(i - j)
                attention_matrix[i, j] *= np.exp(-distance / 5)
        
        # Normalizar
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Crear heatmap
        im = ax.imshow(attention_matrix, cmap='plasma', aspect='auto', interpolation='bilinear')
        
        # Configurar ejes
        ax.set_xlabel('Posición Key', color='white', fontsize=12)
        ax.set_ylabel('Posición Query', color='white', fontsize=12)
        ax.set_title('🎯 Mapa de Atención - Transformer', color='white', fontsize=14, weight='bold')
        ax.tick_params(colors='white')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Peso de Atención', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
        
        # Integrar con tkinter
        canvas = FigureCanvasTkinter(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return viz_frame
    
    def add_layer_connections(self, fig, layer_positions):
        """
        Agrega conexiones visuales entre capas
        """
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]
            
            # Conectar algunas neuronas representativas
            num_connections = min(10, len(current_layer['x']), len(next_layer['x']))
            
            for j in range(num_connections):
                # Seleccionar neuronas para conectar
                curr_idx = j % len(current_layer['x'])
                next_idx = j % len(next_layer['x'])
                
                # Crear línea de conexión
                fig.add_trace(go.Scatter3d(
                    x=[current_layer['x'][curr_idx], next_layer['x'][next_idx]],
                    y=[current_layer['y'][curr_idx], next_layer['y'][next_idx]], 
                    z=[current_layer['z'][curr_idx], next_layer['z'][next_idx]],
                    mode='lines',
                    line=dict(
                        color='rgba(255,255,255,0.3)',
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    def create_training_metrics_3d(self, training_history=None):
        """
        Visualización 3D de métricas de entrenamiento
        """
        if training_history is None:
            # Datos simulados
            epochs = np.arange(1, 51)
            train_acc = 0.5 + 0.45 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.02, 50)
            val_acc = 0.5 + 0.4 * (1 - np.exp(-epochs/12)) + np.random.normal(0, 0.03, 50)
            train_loss = 2 * np.exp(-epochs/8) + np.random.normal(0, 0.05, 50)
            val_loss = 2.2 * np.exp(-epochs/10) + np.random.normal(0, 0.07, 50)
            learning_rate = 0.001 * np.exp(-epochs/20)
        else:
            epochs = training_history['epochs']
            train_acc = training_history['train_acc']
            val_acc = training_history['val_acc']
            train_loss = training_history['train_loss']
            val_loss = training_history['val_loss']
            learning_rate = training_history['learning_rate']
        
        # Crear subplots 3D
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy 3D', 'Loss 3D', 'Learning Rate 3D', 'Métricas Combinadas'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Accuracy 3D
        fig.add_trace(
            go.Scatter3d(
                x=epochs,
                y=train_acc,
                z=val_acc,
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color=epochs,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                line=dict(color=self.colors['primary'], width=4),
                name='Accuracy Evolution'
            ),
            row=1, col=1
        )
        
        # Loss 3D
        fig.add_trace(
            go.Scatter3d(
                x=epochs,
                y=train_loss,
                z=val_loss,
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color=epochs,
                    colorscale='Plasma',
                    opacity=0.8
                ),
                line=dict(color=self.colors['error'], width=4),
                name='Loss Evolution'
            ),
            row=1, col=2
        )
        
        # Learning Rate 3D
        fig.add_trace(
            go.Scatter3d(
                x=epochs,
                y=learning_rate,
                z=np.zeros_like(epochs),
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=learning_rate,
                    colorscale='Cividis',
                    opacity=0.9
                ),
                line=dict(color=self.colors['neural'], width=4),
                name='Learning Rate'
            ),
            row=2, col=1
        )
        
        # Métricas combinadas
        fig.add_trace(
            go.Scatter3d(
                x=train_acc,
                y=val_acc,
                z=train_loss,
                mode='markers',
                marker=dict(
                    size=8,
                    color=epochs,
                    colorscale='Rainbow',
                    opacity=0.8,
                    colorbar=dict(title="Época")
                ),
                name='Métricas Combinadas',
                text=[f"Época {e}" for e in epochs],
                hovertemplate="<b>Época %{text}</b><br>Train Acc: %{x:.3f}<br>Val Acc: %{y:.3f}<br>Train Loss: %{z:.3f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "📊 Métricas de Entrenamiento 3D Holográficas",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(bgcolor="rgba(0,0,0,0.9)"),
            scene2=dict(bgcolor="rgba(0,0,0,0.9)"),
            scene3=dict(bgcolor="rgba(0,0,0,0.9)"),
            scene4=dict(bgcolor="rgba(0,0,0,0.9)"),
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white"),
            showlegend=True
        )
        
        return fig
    
    def create_data_exploration_3d(self, data, labels=None):
        """
        Exploración 3D de datos usando t-SNE y PCA
        """
        if data is None:
            # Datos simulados
            n_samples = 1000
            n_features = 50
            data = np.random.randn(n_samples, n_features)
            labels = np.random.randint(0, 5, n_samples)
        
        # Reducción de dimensionalidad
        if data.shape[1] > 3:
            # t-SNE para visualización
            tsne = TSNE(n_components=3, random_state=42, perplexity=30)
            data_tsne = tsne.fit_transform(data)
            
            # PCA para comparación
            pca = PCA(n_components=3)
            data_pca = pca.fit_transform(data)
        else:
            data_tsne = data
            data_pca = data
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('t-SNE 3D', 'PCA 3D'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Colores para las clases
        colors = px.colors.qualitative.Set1
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                
                # t-SNE plot
                fig.add_trace(
                    go.Scatter3d(
                        x=data_tsne[mask, 0],
                        y=data_tsne[mask, 1],
                        z=data_tsne[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        name=f'Clase {label}',
                        legendgroup=f'clase_{label}'
                    ),
                    row=1, col=1
                )
                
                # PCA plot
                fig.add_trace(
                    go.Scatter3d(
                        x=data_pca[mask, 0],
                        y=data_pca[mask, 1],
                        z=data_pca[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        name=f'Clase {label}',
                        legendgroup=f'clase_{label}',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        else:
            # Sin etiquetas
            fig.add_trace(
                go.Scatter3d(
                    x=data_tsne[:, 0],
                    y=data_tsne[:, 1],
                    z=data_tsne[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=data_tsne[:, 0],
                        colorscale='Viridis',
                        opacity=0.7
                    ),
                    name='t-SNE'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=data_pca[:, 0],
                    y=data_pca[:, 1],
                    z=data_pca[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=data_pca[:, 0],
                        colorscale='Plasma',
                        opacity=0.7
                    ),
                    name='PCA'
                ),
                row=1, col=2
            )
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "🔍 Exploración de Datos 3D Holográfica",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(
                bgcolor="rgba(0,0,0,0.9)",
                xaxis=dict(title="Componente 1"),
                yaxis=dict(title="Componente 2"),
                zaxis=dict(title="Componente 3")
            ),
            scene2=dict(
                bgcolor="rgba(0,0,0,0.9)",
                xaxis=dict(title="PC1"),
                yaxis=dict(title="PC2"),
                zaxis=dict(title="PC3")
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white"),
            showlegend=True
        )
        
        return fig
    
    def create_realtime_training_animation(self):
        """
        Animación en tiempo real del entrenamiento
        """
        # Datos iniciales
        epochs = []
        accuracies = []
        losses = []
        
        fig = go.Figure()
        
        # Configurar trazas iniciales
        fig.add_trace(go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode='markers+lines',
            marker=dict(size=8, color=self.colors['primary']),
            line=dict(color=self.colors['primary'], width=4),
            name='Training Progress'
        ))
        
        # Layout
        fig.update_layout(
            title="🚀 Entrenamiento en Tiempo Real",
            scene=dict(
                xaxis=dict(title="Época"),
                yaxis=dict(title="Accuracy"),
                zaxis=dict(title="Loss"),
                bgcolor="rgba(0,0,0,0.9)"
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white")
        )
        
        return fig
    
    def create_model_comparison_3d(self, models_data):
        """
        Comparación 3D de múltiples modelos
        """
        if models_data is None:
            # Datos simulados de comparación
            models_data = {
                'CNN': {'accuracy': 0.95, 'loss': 0.15, 'params': 1000000, 'time': 120},
                'RNN': {'accuracy': 0.88, 'loss': 0.25, 'params': 500000, 'time': 180},
                'GAN': {'accuracy': 0.92, 'loss': 0.18, 'params': 2000000, 'time': 300},
                'VAE': {'accuracy': 0.85, 'loss': 0.30, 'params': 800000, 'time': 200},
                'Transformer': {'accuracy': 0.97, 'loss': 0.12, 'params': 5000000, 'time': 400}
            }
        
        fig = go.Figure()
        
        model_names = list(models_data.keys())
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['neural'], self.colors['success']]
        
        for i, (model_name, data) in enumerate(models_data.items()):
            fig.add_trace(go.Scatter3d(
                x=[data['accuracy']],
                y=[data['loss']],
                z=[data['params'] / 1000000],  # En millones
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=[model_name],
                textposition="top center",
                name=model_name,
                hovertemplate=f"<b>{model_name}</b><br>Accuracy: %{{x:.3f}}<br>Loss: %{{y:.3f}}<br>Parámetros: %{{z:.1f}}M<br>Tiempo: {data['time']}s<extra></extra>"
            ))
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "⚖️ Comparación de Modelos 3D",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(
                xaxis=dict(title="Accuracy", range=[0.8, 1.0]),
                yaxis=dict(title="Loss", range=[0.1, 0.35]),
                zaxis=dict(title="Parámetros (Millones)"),
                bgcolor="rgba(0,0,0,0.9)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white"),
            showlegend=True
        )
        
        return fig
    
    def create_hyperparameter_space_3d(self, hyperparams_data=None):
        """
        Visualización 3D del espacio de hiperparámetros
        """
        if hyperparams_data is None:
            # Datos simulados
            n_points = 100
            learning_rates = np.random.uniform(0.0001, 0.01, n_points)
            batch_sizes = np.random.choice([16, 32, 64, 128], n_points)
            dropouts = np.random.uniform(0.1, 0.5, n_points)
            accuracies = (0.8 + 0.15 * np.random.random(n_points) + 
                         0.05 * (learning_rates / 0.01) - 
                         0.03 * (dropouts - 0.3)**2)
        else:
            learning_rates = hyperparams_data['learning_rates']
            batch_sizes = hyperparams_data['batch_sizes']
            dropouts = hyperparams_data['dropouts']
            accuracies = hyperparams_data['accuracies']
        
        fig = go.Figure()
        
        # Scatter 3D con colores basados en accuracy
        fig.add_trace(go.Scatter3d(
            x=learning_rates,
            y=batch_sizes,
            z=dropouts,
            mode='markers',
            marker=dict(
                size=8,
                color=accuracies,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Accuracy",
                    titleside="right",
                    titlefont=dict(color="white")
                ),
                line=dict(width=1, color='white')
            ),
            text=[f"LR: {lr:.4f}<br>BS: {bs}<br>Dropout: {d:.2f}<br>Acc: {acc:.3f}" 
                  for lr, bs, d, acc in zip(learning_rates, batch_sizes, dropouts, accuracies)],
            hovertemplate="<b>Configuración</b><br>%{text}<extra></extra>",
            name="Hiperparámetros"
        ))
        
        # Encontrar el mejor punto
        best_idx = np.argmax(accuracies)
        fig.add_trace(go.Scatter3d(
            x=[learning_rates[best_idx]],
            y=[batch_sizes[best_idx]],
            z=[dropouts[best_idx]],
            mode='markers',
            marker=dict(
                size=15,
                color=self.colors['success'],
                symbol='diamond',
                line=dict(width=3, color='white')
            ),
            name="Mejor Configuración",
            hovertemplate=f"<b>Mejor Configuración</b><br>LR: {learning_rates[best_idx]:.4f}<br>BS: {batch_sizes[best_idx]}<br>Dropout: {dropouts[best_idx]:.2f}<br>Acc: {accuracies[best_idx]:.3f}<extra></extra>"
        ))
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "🎯 Espacio de Hiperparámetros 3D",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(
                xaxis=dict(title="Learning Rate", type="log"),
                yaxis=dict(title="Batch Size"),
                zaxis=dict(title="Dropout Rate"),
                bgcolor="rgba(0,0,0,0.9)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            plot_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white"),
            showlegend=True
        )
        
        return fig
    
    def animate_training_progress(self, fig, epoch_data):
        """
        Anima el progreso del entrenamiento en tiempo real
        """
        # Actualizar datos de la figura
        with fig.batch_update():
            fig.data[0].x = epoch_data['epochs']
            fig.data[0].y = epoch_data['accuracies']
            fig.data[0].z = epoch_data['losses']
        
        return fig
    
    def create_attention_heatmap_3d(self, attention_weights=None):
        """
        Visualización 3D de mapas de atención
        """
        if attention_weights is None:
            # Simular pesos de atención
            seq_len = 20
            attention_weights = np.random.random((seq_len, seq_len))
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        fig = go.Figure()
        
        # Crear superficie 3D
        x = np.arange(attention_weights.shape[0])
        y = np.arange(attention_weights.shape[1])
        X, Y = np.meshgrid(x, y)
        
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=attention_weights,
            colorscale='Viridis',
            opacity=0.8,
            name="Attention Weights"
        ))
        
        # Configuración del layout
        fig.update_layout(
            title={
                'text': "🎯 Mapa de Atención 3D",
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['primary']}
            },
            scene=dict(
                xaxis=dict(title="Posición Query"),
                yaxis=dict(title="Posición Key"),
                zaxis=dict(title="Peso de Atención"),
                bgcolor="rgba(0,0,0,0.9)"
            ),
            paper_bgcolor="rgba(0,0,0,0.9)",
            font=dict(color="white")
        )
        
        return fig