"""
Sistema RAG (Retrieval-Augmented Generation) Autónomo
Implementa recuperación de información y generación aumentada para el ecosistema
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import os
from pathlib import Path
import pickle
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Librerías para embeddings y búsqueda
from sentence_transformers import SentenceTransformer
import faiss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    pipeline, BitsAndBytesConfig
)

# Librerías para procesamiento de documentos
import PyPDF2
import docx
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import sqlite3

# Librerías para análisis de texto
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Procesador de documentos para extraer y preparar contenido para RAG"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx', '.html', '.json', '.csv']
        
        # Inicializar herramientas de NLP
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("NLTK no disponible, usando procesamiento básico")
            self.lemmatizer = None
            self.stop_words = set()
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("SpaCy no disponible, usando procesamiento básico")
            self.nlp = None
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extraer texto de diferentes tipos de archivos"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return self._extract_from_txt(file_path)
        elif extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension == '.docx':
            return self._extract_from_docx(file_path)
        elif extension == '.html':
            return self._extract_from_html(file_path)
        elif extension == '.json':
            return self._extract_from_json(file_path)
        elif extension == '.csv':
            return self._extract_from_csv(file_path)
        else:
            raise ValueError(f"Formato no soportado: {extension}")
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extraer texto de archivo TXT"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extraer texto de archivo PDF"""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extraer texto de archivo DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except:
            logger.error(f"Error procesando DOCX: {file_path}")
            return ""
    
    def _extract_from_html(self, file_path: Path) -> str:
        """Extraer texto de archivo HTML"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
    
    def _extract_from_json(self, file_path: Path) -> str:
        """Extraer texto de archivo JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    
    def _extract_from_csv(self, file_path: Path) -> str:
        """Extraer texto de archivo CSV"""
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def extract_from_url(self, url: str) -> str:
        """Extraer texto de una URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remover scripts y estilos
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text()
        except Exception as e:
            logger.error(f"Error extrayendo de URL {url}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Dividir texto en chunks para procesamiento"""
        if self.nlp:
            # Usar spaCy para división inteligente
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # División básica por oraciones
            sentences = sent_tokenize(text) if 'sent_tokenize' in globals() else text.split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Agregar overlap entre chunks
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Agregar overlap del chunk anterior
                    prev_words = chunks[i-1].split()[-overlap:]
                    overlapped_chunk = " ".join(prev_words) + " " + chunk
                    overlapped_chunks.append(overlapped_chunk)
            chunks = overlapped_chunks
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesar texto para mejorar la búsqueda"""
        # Limpiar texto básico
        text = text.strip()
        text = ' '.join(text.split())  # Normalizar espacios
        
        if self.nlp:
            # Usar spaCy para preprocesamiento avanzado
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text.strip():
                    tokens.append(token.lemma_.lower())
            return ' '.join(tokens)
        else:
            # Preprocesamiento básico
            words = word_tokenize(text.lower()) if 'word_tokenize' in globals() else text.lower().split()
            if self.lemmatizer and self.stop_words:
                words = [self.lemmatizer.lemmatize(word) for word in words 
                        if word not in self.stop_words and word.isalpha()]
            return ' '.join(words)

class VectorStore:
    """Almacén de vectores para búsqueda semántica eficiente"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 index_type: str = "flat", dimension: int = 384):
        
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        
        # Cargar modelo de embeddings
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            self.embedding_model = None
        
        # Crear índice FAISS
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Almacenar metadatos
        self.documents = []
        self.metadata = []
        self.is_trained = False
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Agregar documentos al almacén de vectores"""
        if not self.embedding_model:
            logger.error("Modelo de embeddings no disponible")
            return
        
        if metadata is None:
            metadata = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        
        # Generar embeddings
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalizar para cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Entrenar índice si es necesario
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings)
            self.is_trained = True
        
        # Agregar al índice
        self.index.add(embeddings)
        
        # Almacenar documentos y metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Agregados {len(documents)} documentos al vector store")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Buscar documentos similares a la consulta"""
        if not self.embedding_model:
            logger.error("Modelo de embeddings no disponible")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Vector store vacío")
            return []
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Buscar
        scores, indices = self.index.search(query_embedding, k)
        
        # Filtrar por threshold y preparar resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                }
                results.append(result)
        
        return results
    
    def save(self, path: str):
        """Guardar vector store en disco"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar índice FAISS
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Guardar metadatos
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_model_name': self.embedding_model_name,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Vector store guardado en {path}")
    
    def load(self, path: str):
        """Cargar vector store desde disco"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store no encontrado en {path}")
        
        # Cargar índice FAISS
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Cargar metadatos
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_model_name = data['embedding_model_name']
            self.dimension = data['dimension']
        
        logger.info(f"Vector store cargado desde {path}")

class RAGGenerator:
    """Generador RAG que combina recuperación y generación"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Configuración del modelo generativo
        self.model_name = config.get('generator_model', 'microsoft/DialoGPT-medium')
        self.max_length = config.get('max_length', 512)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        
        # Cargar modelo generativo
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Configurar pad token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error cargando modelo generativo: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, query: str, context_documents: List[str], 
                         max_context_length: int = 1000) -> str:
        """Generar respuesta usando RAG"""
        
        if not self.model or not self.tokenizer:
            return "Error: Modelo generativo no disponible"
        
        # Preparar contexto
        context = self._prepare_context(context_documents, max_context_length)
        
        # Crear prompt
        prompt = self._create_prompt(query, context)
        
        # Tokenizar
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, 
                                     max_length=self.max_length)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la parte nueva (respuesta)
        response = response[len(prompt):].strip()
        
        return response
    
    def _prepare_context(self, documents: List[str], max_length: int) -> str:
        """Preparar contexto combinando documentos relevantes"""
        context = ""
        current_length = 0
        
        for doc in documents:
            if current_length + len(doc) <= max_length:
                context += doc + "\n\n"
                current_length += len(doc)
            else:
                # Agregar parte del documento que quepa
                remaining = max_length - current_length
                if remaining > 100:  # Solo si queda espacio significativo
                    context += doc[:remaining] + "..."
                break
        
        return context.strip()
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Crear prompt para el modelo generativo"""
        prompt_template = self.config.get('prompt_template', 
            "Contexto: {context}\n\nPregunta: {query}\n\nRespuesta:")
        
        return prompt_template.format(context=context, query=query)

class AutonomousRAGSystem:
    """Sistema RAG autónomo con capacidades de auto-mejora"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = config.get('data_sources', [])
        self.update_interval = config.get('update_interval', 3600)  # 1 hora
        self.auto_update = config.get('auto_update', True)
        
        # Componentes del sistema
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(
            embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2'),
            index_type=config.get('index_type', 'flat')
        )
        self.generator = RAGGenerator(config)
        
        # Estado del sistema
        self.last_update = None
        self.performance_metrics = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'average_response_time': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Base de datos para logging
        self.db_path = config.get('db_path', 'rag_system.db')
        self._init_database()
        
        # Hilo para actualizaciones automáticas
        self.update_thread = None
        self.stop_updates = threading.Event()
        
        if self.auto_update:
            self.start_auto_updates()
    
    def _init_database(self):
        """Inicializar base de datos para logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                context_docs INTEGER,
                response_time REAL,
                user_rating INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content_hash TEXT,
                last_updated TEXT,
                chunk_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_data_source(self, source: str, source_type: str = 'auto'):
        """Agregar nueva fuente de datos"""
        if source not in self.data_sources:
            self.data_sources.append(source)
            self._process_data_source(source, source_type)
            logger.info(f"Nueva fuente de datos agregada: {source}")
    
    def _process_data_source(self, source: str, source_type: str = 'auto'):
        """Procesar una fuente de datos"""
        try:
            if source_type == 'auto':
                # Detectar tipo automáticamente
                if source.startswith(('http://', 'https://')):
                    source_type = 'url'
                elif os.path.isfile(source):
                    source_type = 'file'
                elif os.path.isdir(source):
                    source_type = 'directory'
                else:
                    source_type = 'text'
            
            documents = []
            metadata = []
            
            if source_type == 'file':
                text = self.document_processor.extract_text_from_file(source)
                chunks = self.document_processor.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'source': source,
                        'chunk_id': i,
                        'source_type': 'file'
                    })
            
            elif source_type == 'directory':
                for root, dirs, files in os.walk(source):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            text = self.document_processor.extract_text_from_file(file_path)
                            chunks = self.document_processor.chunk_text(text)
                            
                            for i, chunk in enumerate(chunks):
                                documents.append(chunk)
                                metadata.append({
                                    'source': file_path,
                                    'chunk_id': i,
                                    'source_type': 'file'
                                })
                        except Exception as e:
                            logger.warning(f"Error procesando {file_path}: {e}")
            
            elif source_type == 'url':
                text = self.document_processor.extract_from_url(source)
                chunks = self.document_processor.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'source': source,
                        'chunk_id': i,
                        'source_type': 'url'
                    })
            
            elif source_type == 'text':
                chunks = self.document_processor.chunk_text(source)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'source': 'direct_text',
                        'chunk_id': i,
                        'source_type': 'text'
                    })
            
            # Agregar al vector store
            if documents:
                self.vector_store.add_documents(documents, metadata)
                
                # Registrar en base de datos
                self._log_documents(source, len(documents))
                
                logger.info(f"Procesados {len(documents)} chunks de {source}")
            
        except Exception as e:
            logger.error(f"Error procesando fuente {source}: {e}")
    
    def query(self, question: str, k: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
        """Realizar consulta RAG"""
        start_time = time.time()
        
        try:
            # Buscar documentos relevantes
            search_results = self.vector_store.search(question, k=k, threshold=threshold)
            
            if not search_results:
                response = "Lo siento, no encontré información relevante para responder tu pregunta."
                context_docs = 0
            else:
                # Extraer documentos para contexto
                context_documents = [result['document'] for result in search_results]
                
                # Generar respuesta
                response = self.generator.generate_response(question, context_documents)
                context_docs = len(context_documents)
            
            response_time = time.time() - start_time
            
            # Actualizar métricas
            self._update_metrics(response_time, len(search_results) > 0)
            
            # Registrar consulta
            self._log_query(question, response, context_docs, response_time)
            
            return {
                'question': question,
                'response': response,
                'context_documents': search_results,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en consulta RAG: {e}")
            return {
                'question': question,
                'response': f"Error procesando consulta: {str(e)}",
                'context_documents': [],
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_metrics(self, response_time: float, successful_retrieval: bool):
        """Actualizar métricas de rendimiento"""
        self.performance_metrics['total_queries'] += 1
        
        if successful_retrieval:
            self.performance_metrics['successful_retrievals'] += 1
        
        # Actualizar tiempo promedio de respuesta
        total_queries = self.performance_metrics['total_queries']
        current_avg = self.performance_metrics['average_response_time']
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_metrics['average_response_time'] = new_avg
    
    def _log_query(self, query: str, response: str, context_docs: int, response_time: float):
        """Registrar consulta en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO queries (timestamp, query, response, context_docs, response_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), query, response, context_docs, response_time))
        
        conn.commit()
        conn.close()
    
    def _log_documents(self, source: str, chunk_count: int):
        """Registrar documentos procesados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calcular hash del contenido (simplificado)
        content_hash = str(hash(source + str(chunk_count)))
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents (source, content_hash, last_updated, chunk_count)
            VALUES (?, ?, ?, ?)
        ''', (source, content_hash, datetime.now().isoformat(), chunk_count))
        
        conn.commit()
        conn.close()
    
    def start_auto_updates(self):
        """Iniciar actualizaciones automáticas"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._auto_update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Actualizaciones automáticas iniciadas")
    
    def stop_auto_updates(self):
        """Detener actualizaciones automáticas"""
        self.stop_updates.set()
        if self.update_thread:
            self.update_thread.join()
        logger.info("Actualizaciones automáticas detenidas")
    
    def _auto_update_loop(self):
        """Loop de actualizaciones automáticas"""
        while not self.stop_updates.wait(self.update_interval):
            try:
                self.update_knowledge_base()
            except Exception as e:
                logger.error(f"Error en actualización automática: {e}")
    
    def update_knowledge_base(self):
        """Actualizar base de conocimiento"""
        logger.info("Iniciando actualización de base de conocimiento...")
        
        for source in self.data_sources:
            try:
                self._process_data_source(source)
            except Exception as e:
                logger.error(f"Error actualizando fuente {source}: {e}")
        
        self.last_update = datetime.now()
        logger.info("Actualización de base de conocimiento completada")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de rendimiento"""
        success_rate = 0.0
        if self.performance_metrics['total_queries'] > 0:
            success_rate = (self.performance_metrics['successful_retrievals'] / 
                          self.performance_metrics['total_queries']) * 100
        
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'successful_retrievals': self.performance_metrics['successful_retrievals'],
            'success_rate': success_rate,
            'average_response_time': self.performance_metrics['average_response_time'],
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'data_sources_count': len(self.data_sources),
            'vector_store_size': self.vector_store.index.ntotal
        }
    
    def save_system_state(self, path: str):
        """Guardar estado del sistema"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar vector store
        self.vector_store.save(path / "vector_store")
        
        # Guardar configuración y métricas
        system_state = {
            'config': self.config,
            'data_sources': self.data_sources,
            'performance_metrics': self.performance_metrics,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        with open(path / "system_state.json", 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logger.info(f"Estado del sistema guardado en {path}")
    
    def load_system_state(self, path: str):
        """Cargar estado del sistema"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Estado del sistema no encontrado en {path}")
        
        # Cargar vector store
        self.vector_store.load(path / "vector_store")
        
        # Cargar configuración y métricas
        with open(path / "system_state.json", 'r') as f:
            system_state = json.load(f)
            
            self.data_sources = system_state.get('data_sources', [])
            self.performance_metrics = system_state.get('performance_metrics', {})
            
            last_update_str = system_state.get('last_update')
            if last_update_str:
                self.last_update = datetime.fromisoformat(last_update_str)
        
        logger.info(f"Estado del sistema cargado desde {path}")

def create_rag_system(config: Dict[str, Any] = None) -> AutonomousRAGSystem:
    """Factory function para crear sistema RAG"""
    
    default_config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'generator_model': 'microsoft/DialoGPT-medium',
        'index_type': 'flat',
        'max_length': 512,
        'temperature': 0.7,
        'top_p': 0.9,
        'update_interval': 3600,
        'auto_update': True,
        'data_sources': [],
        'db_path': 'rag_system.db'
    }
    
    if config:
        default_config.update(config)
    
    return AutonomousRAGSystem(default_config)

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔍 Creando Sistema RAG Autónomo...")
    
    # Configuración del sistema
    config = {
        'data_sources': ['./data/'],
        'embedding_model': 'all-MiniLM-L6-v2',
        'generator_model': 'microsoft/DialoGPT-medium',
        'auto_update': True
    }
    
    # Crear sistema RAG
    rag_system = create_rag_system(config)
    
    # Agregar fuentes de datos
    rag_system.add_data_source('./data/sample.txt', 'file')
    
    # Realizar consulta
    result = rag_system.query("¿Qué información importante contienen los datos?")
    print(f"Respuesta: {result['response']}")
    
    # Obtener reporte de rendimiento
    report = rag_system.get_performance_report()
    print(f"Reporte: {report}")
    
    print("Sistema RAG Autónomo creado exitosamente! 🚀")