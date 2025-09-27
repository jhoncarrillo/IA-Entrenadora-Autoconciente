import PyPDF2
import pandas as pd
import sqlite3
import json
import requests
import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Union, Optional
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class AdvancedDataExtractor:
    def __init__(self):
        # Descargar recursos de NLTK si no existen
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def extract_from_txt(self, file_path: str, encoding: str = 'utf-8') -> Dict[str, Union[str, List[str], Dict]]:
        """Extracción avanzada de archivos de texto con análisis"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # Análisis básico del texto
            sentences = sent_tokenize(content)
            words = word_tokenize(content.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            
            return {
                'raw_content': content,
                'sentences': sentences,
                'word_count': len(words),
                'unique_words': len(set(words)),
                'filtered_words': filtered_words[:100],  # Top 100 palabras relevantes
                'sentence_count': len(sentences),
                'metadata': {
                    'file_size': os.path.getsize(file_path),
                    'encoding': encoding
                }
            }
        except Exception as e:
            return {'error': str(e), 'raw_content': ''}

    def extract_from_csv(self, file_path: str) -> Dict[str, Union[List, Dict]]:
        """Extracción avanzada de CSV con análisis estadístico"""
        try:
            df = pd.read_csv(file_path)
            
            return {
                'data': df.to_dict(orient='records'),
                'summary': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'null_counts': df.isnull().sum().to_dict(),
                    'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                },
                'sample_data': df.head(5).to_dict(orient='records')
            }
        except Exception as e:
            return {'error': str(e), 'data': []}

    def extract_from_json(self, file_path: str) -> Dict:
        """Extracción de JSON con análisis de estructura"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            def analyze_structure(obj, depth=0, max_depth=3):
                if depth > max_depth:
                    return "..."
                
                if isinstance(obj, dict):
                    return {k: analyze_structure(v, depth+1, max_depth) for k, v in list(obj.items())[:5]}
                elif isinstance(obj, list):
                    return [analyze_structure(item, depth+1, max_depth) for item in obj[:3]]
                else:
                    return type(obj).__name__
            
            return {
                'data': data,
                'structure_analysis': analyze_structure(data),
                'metadata': {
                    'type': type(data).__name__,
                    'size': len(str(data)),
                    'keys': list(data.keys()) if isinstance(data, dict) else None,
                    'length': len(data) if isinstance(data, (list, dict)) else None
                }
            }
        except Exception as e:
            return {'error': str(e), 'data': {}}

    def extract_from_pdf(self, file_path: str) -> Dict[str, Union[str, List, Dict]]:
        """Extracción avanzada de PDF con análisis de contenido"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                pages_content = []
                full_text = ""
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    pages_content.append({
                        'page_number': i + 1,
                        'content': page_text,
                        'word_count': len(page_text.split())
                    })
                    full_text += page_text + "\n"
                
                # Análisis del contenido
                sentences = sent_tokenize(full_text)
                words = word_tokenize(full_text.lower())
                
                return {
                    'full_text': full_text,
                    'pages': pages_content,
                    'metadata': {
                        'total_pages': len(pdf_reader.pages),
                        'total_words': len(words),
                        'total_sentences': len(sentences),
                        'file_size': os.path.getsize(file_path)
                    },
                    'summary': {
                        'first_100_words': ' '.join(words[:100]),
                        'avg_words_per_page': len(words) / len(pdf_reader.pages) if pdf_reader.pages else 0
                    }
                }
        except Exception as e:
            return {'error': str(e), 'full_text': '', 'pages': []}

    def extract_from_db(self, file_path: str, query: str = "SELECT * FROM employees", table_name: Optional[str] = None) -> Dict:
        """Extracción avanzada de base de datos con análisis de esquema"""
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Si no se proporciona query, intentar obtener todas las tablas
            if table_name is None:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                table_name = tables[0][0] if tables else 'employees'
                query = f"SELECT * FROM {table_name} LIMIT 100"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            # Análisis del esquema
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema_info = cursor.fetchall()
            
            conn.close()
            
            return {
                'data': [dict(zip(columns, row)) for row in rows],
                'schema': {
                    'columns': columns,
                    'column_info': [{'name': col[1], 'type': col[2], 'not_null': col[3]} for col in schema_info],
                    'total_rows': len(rows)
                },
                'metadata': {
                    'table_name': table_name,
                    'query_used': query,
                    'file_size': os.path.getsize(file_path)
                }
            }
        except Exception as e:
            return {'error': str(e), 'data': [], 'schema': {}}

    def extract_from_image(self, file_path: str) -> Dict:
        """Extracción de información de imágenes"""
        try:
            # Cargar imagen con OpenCV
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("No se pudo cargar la imagen")
            
            # Información básica
            height, width, channels = img.shape
            
            # Análisis de color
            mean_color = np.mean(img, axis=(0, 1))
            
            # Detectar bordes
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.sum(edges > 0)
            
            return {
                'metadata': {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'file_size': os.path.getsize(file_path),
                    'format': file_path.split('.')[-1].upper()
                },
                'analysis': {
                    'mean_color_bgr': mean_color.tolist(),
                    'edge_density': edge_count / (width * height),
                    'brightness': np.mean(gray),
                    'contrast': np.std(gray)
                }
            }
        except Exception as e:
            return {'error': str(e), 'metadata': {}, 'analysis': {}}

    def extract_from_url(self, url: str) -> Dict:
        """Extracción de contenido web"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer texto
            text = soup.get_text()
            sentences = sent_tokenize(text)
            
            # Extraer enlaces
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            
            # Extraer imágenes
            images = [img.get('src') for img in soup.find_all('img', src=True)]
            
            return {
                'content': text[:5000],  # Primeros 5000 caracteres
                'metadata': {
                    'title': soup.title.string if soup.title else '',
                    'url': url,
                    'status_code': response.status_code,
                    'content_length': len(text),
                    'sentence_count': len(sentences)
                },
                'links': links[:20],  # Primeros 20 enlaces
                'images': images[:10],  # Primeras 10 imágenes
                'headers': dict(response.headers)
            }
        except Exception as e:
            return {'error': str(e), 'content': '', 'metadata': {}}

# Funciones de compatibilidad
def extract_from_txt(file_path: str) -> str:
    extractor = AdvancedDataExtractor()
    result = extractor.extract_from_txt(file_path)
    return result.get('raw_content', '')

def extract_from_csv(file_path: str) -> List[Dict]:
    extractor = AdvancedDataExtractor()
    result = extractor.extract_from_csv(file_path)
    return result.get('data', [])

def extract_from_json(file_path: str) -> Dict:
    extractor = AdvancedDataExtractor()
    result = extractor.extract_from_json(file_path)
    return result.get('data', {})

def extract_from_pdf(file_path: str) -> str:
    extractor = AdvancedDataExtractor()
    result = extractor.extract_from_pdf(file_path)
    return result.get('full_text', '')

def extract_from_db(file_path: str, query: str = "SELECT * FROM employees") -> List[Dict]:
    extractor = AdvancedDataExtractor()
    result = extractor.extract_from_db(file_path, query)
    return result.get('data', [])

def extract_data(source: Dict) -> Union[str, List, Dict]:
    """Función principal de extracción de datos mejorada"""
    extractor = AdvancedDataExtractor()
    
    if source['type'] == 'txt':
        return extractor.extract_from_txt(source['path'])
    elif source['type'] == 'csv':
        return extractor.extract_from_csv(source['path'])
    elif source['type'] == 'json':
        return extractor.extract_from_json(source['path'])
    elif source['type'] == 'pdf':
        return extractor.extract_from_pdf(source['path'])
    elif source['type'] == 'db':
        return extractor.extract_from_db(source['path'])
    elif source['type'] == 'image':
        return extractor.extract_from_image(source['path'])
    elif source['type'] == 'url':
        return extractor.extract_from_url(source['path'])
    else:
        raise ValueError(f"Unsupported data type: {source['type']}")

# Alias para compatibilidad con ecosystem_training.py
extract_from_text = extract_from_txt
extract_from_sql = extract_from_db
