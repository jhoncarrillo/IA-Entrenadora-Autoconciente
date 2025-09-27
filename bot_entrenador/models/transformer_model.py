"""
Modelo Transformer Avanzado para el Ecosistema Autónomo
Implementa arquitectura Transformer con capacidades de Fine-Tuning y RAG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertForSequenceClassification, GPT2LMHeadModel,
    T5ForConditionalGeneration, RobertaModel
)
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Implementación personalizada de atención multi-cabeza"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(context)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    """Bloque Transformer con normalización y conexiones residuales"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class AdvancedTransformer(nn.Module):
    """Transformer Avanzado con capacidades de Fine-Tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        self.d_model = config.get('d_model', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 2048)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 2)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Inicialización de pesos"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, task='classification'):
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout_layer(x)
        
        # Apply transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, attention_mask)
            attention_weights.append(attn_weights)
        
        # Final normalization
        x = self.norm(x)
        
        if task == 'classification':
            # Use [CLS] token (first token) for classification
            cls_output = x[:, 0, :]
            logits = self.classifier(cls_output)
            return {
                'logits': logits,
                'hidden_states': x,
                'attention_weights': attention_weights
            }
        elif task == 'generation':
            # Language modeling head
            logits = self.lm_head(x)
            return {
                'logits': logits,
                'hidden_states': x,
                'attention_weights': attention_weights
            }
        else:
            return {
                'hidden_states': x,
                'attention_weights': attention_weights
            }

class HybridTransformer(nn.Module):
    """Transformer Híbrido que combina modelos preentrenados con arquitectura personalizada"""
    
    def __init__(self, pretrained_model_name: str = "bert-base-uncased", 
                 custom_layers: int = 2, freeze_pretrained: bool = False):
        super().__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.freeze_pretrained = freeze_pretrained
        
        # Cargar modelo preentrenado
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.pretrained_model.config
        
        # Congelar parámetros del modelo preentrenado si se especifica
        if freeze_pretrained:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Capas personalizadas adicionales
        self.custom_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                d_ff=self.config.intermediate_size,
                dropout=self.config.hidden_dropout_prob
            )
            for _ in range(custom_layers)
        ])
        
        # Capas de salida adaptables
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.regression_head = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, task='classification'):
        # Obtener representaciones del modelo preentrenado
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Aplicar capas personalizadas
        attention_weights = []
        for custom_layer in self.custom_layers:
            hidden_states, attn_weights = custom_layer(hidden_states, attention_mask)
            attention_weights.append(attn_weights)
        
        if task == 'classification':
            # Pooling y clasificación
            pooled_output = hidden_states[:, 0, :]  # [CLS] token
            logits = self.classifier(pooled_output)
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'attention_weights': attention_weights
            }
        elif task == 'regression':
            pooled_output = hidden_states[:, 0, :]
            output = self.regression_head(pooled_output)
            return {
                'predictions': output,
                'hidden_states': hidden_states,
                'attention_weights': attention_weights
            }
        else:
            return {
                'hidden_states': hidden_states,
                'attention_weights': attention_weights
            }

class TransformerFactory:
    """Factory para crear diferentes tipos de transformers"""
    
    @staticmethod
    def create_transformer(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Crear transformer basado en el tipo especificado"""
        
        if model_type == 'custom':
            return AdvancedTransformer(config)
        
        elif model_type == 'hybrid':
            return HybridTransformer(
                pretrained_model_name=config.get('pretrained_model', 'bert-base-uncased'),
                custom_layers=config.get('custom_layers', 2),
                freeze_pretrained=config.get('freeze_pretrained', False)
            )
        
        elif model_type == 'bert':
            return BertForSequenceClassification.from_pretrained(
                config.get('model_name', 'bert-base-uncased'),
                num_labels=config.get('num_classes', 2)
            )
        
        elif model_type == 'gpt2':
            return GPT2LMHeadModel.from_pretrained(
                config.get('model_name', 'gpt2')
            )
        
        elif model_type == 't5':
            return T5ForConditionalGeneration.from_pretrained(
                config.get('model_name', 't5-small')
            )
        
        elif model_type == 'roberta':
            return RobertaModel.from_pretrained(
                config.get('model_name', 'roberta-base')
            )
        
        else:
            raise ValueError(f"Tipo de transformer no soportado: {model_type}")

def create_transformer(model_type: str = 'custom', **kwargs) -> nn.Module:
    """Función de conveniencia para crear transformers"""
    
    default_config = {
        'vocab_size': 50000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 512,
        'dropout': 0.1,
        'num_classes': 2
    }
    
    # Actualizar configuración con argumentos proporcionados
    config = {**default_config, **kwargs}
    
    return TransformerFactory.create_transformer(model_type, config)

def create_adaptive_transformer(task_type: str, data_characteristics: Dict[str, Any]) -> nn.Module:
    """Crear transformer adaptado a las características de los datos"""
    
    # Analizar características de los datos
    vocab_size = data_characteristics.get('vocab_size', 50000)
    max_length = data_characteristics.get('max_sequence_length', 512)
    num_classes = data_characteristics.get('num_classes', 2)
    complexity = data_characteristics.get('complexity_score', 0.5)
    
    # Configuración adaptativa basada en la complejidad
    if complexity < 0.3:  # Datos simples
        config = {
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 3,
            'd_ff': 1024,
            'dropout': 0.1
        }
    elif complexity < 0.7:  # Datos moderados
        config = {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'dropout': 0.15
        }
    else:  # Datos complejos
        config = {
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 12,
            'd_ff': 3072,
            'dropout': 0.2
        }
    
    # Configuración específica de la tarea
    config.update({
        'vocab_size': vocab_size,
        'max_seq_length': max_length,
        'num_classes': num_classes
    })
    
    # Seleccionar tipo de modelo basado en la tarea
    if task_type in ['classification', 'sentiment_analysis']:
        model_type = 'hybrid'
        config['pretrained_model'] = 'bert-base-uncased'
    elif task_type in ['generation', 'text_completion']:
        model_type = 'hybrid'
        config['pretrained_model'] = 'gpt2'
    elif task_type in ['summarization', 'translation']:
        model_type = 't5'
        config['model_name'] = 't5-small'
    else:
        model_type = 'custom'
    
    return create_transformer(model_type, **config)

# Funciones de utilidad para compatibilidad
def create_bert_model(num_labels: int = 2, model_name: str = "bert-base-uncased") -> nn.Module:
    """Crear modelo BERT para clasificación"""
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
