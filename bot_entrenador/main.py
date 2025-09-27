from scripts.data_extraction import extract_data, AdvancedDataExtractor
from scripts.prompt_generation import create_prompt_for_transformers, AdvancedPromptGenerator
from scripts.model_interaction import interact_with_model, AdvancedModelInteraction
from scripts.ecosystem_training import AdvancedEcosystemTraining, train_ecosystem
from scripts.autonomous_ecosystem import AutonomousEcosystem
from scripts.rag_system import create_rag_system, AutonomousRAGSystem
from scripts.fine_tuning_system import create_fine_tuning_system, AutonomousFineTuningSystem

# Importar modelos avanzados
from models import (
    create_model_by_type, create_adaptive_model, get_available_models,
    create_transformer, create_vae, create_diffusion_model, create_gan,
    AVAILABLE_MODELS
)

import os
import logging
import asyncio
import threading
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import sqlite3

# Configurar logging avanzado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('bot_entrenador_advanced.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BotEntrenadorGeneral:
    """Bot Entrenador General - Sistema autónomo de entrenamiento de ecosistemas de IA con RAG y Fine-Tuning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Inicializar componentes básicos
        self.data_extractor = AdvancedDataExtractor()
        self.prompt_generator = AdvancedPromptGenerator()
        self.model_interaction = AdvancedModelInteraction()
        self.ecosystem_trainer = AdvancedEcosystemTraining(self.config.get('training_config'))
        self.autonomous_ecosystem = AutonomousEcosystem(self.config.get('autonomous_config'))
        
        # Inicializar sistemas avanzados
        self.rag_system = create_rag_system(self.config.get('rag_config', {}))
        self.fine_tuning_system = create_fine_tuning_system(self.config.get('fine_tuning_config', {}))
        
        # Estado del sistema
        self.system_state = {
            'initialized_at': datetime.now().isoformat(),
            'active_tasks': {},
            'completed_tasks': [],
            'performance_metrics': {
                'total_models_trained': 0,
                'total_data_processed': 0,
                'total_queries_answered': 0,
                'average_training_time': 0.0,
                'success_rate': 0.0
            }
        }
        
        # Base de datos del sistema
        self.db_path = self.config.get('system_db_path', 'bot_entrenador_system.db')
        self._init_system_database()
        
        # Hilos de procesamiento
        self.task_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.stop_processing = threading.Event()
        
        # Inicializar procesamiento automático
        if self.config.get('auto_start_processing', True):
            self.start_autonomous_processing()
        
        logger.info("Bot Entrenador General Avanzado inicializado exitosamente")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto del bot avanzado"""
        return {
            'data_sources_path': './data',
            'models_path': './models',
            'output_path': './output',
            'system_db_path': 'bot_entrenador_system.db',
            'auto_start_processing': True,
            'max_concurrent_tasks': 4,
            
            # Configuración de entrenamiento
            'training_config': {
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001,
                'save_models': True,
                'use_early_stopping': True,
                'patience': 5
            },
            
            # Configuración del ecosistema autónomo
            'autonomous_config': {
                'auto_discovery': True,
                'continuous_learning': True,
                'performance_threshold': 0.85,
                'auto_optimize': True,
                'max_cycles': 10
            },
            
            # Configuración del sistema RAG
            'rag_config': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'generator_model': 'microsoft/DialoGPT-medium',
                'auto_update': True,
                'update_interval': 3600,
                'data_sources': ['./data/'],
                'max_context_length': 1000
            },
            
            # Configuración del sistema de Fine-Tuning
            'fine_tuning_config': {
                'max_concurrent_trainings': 2,
                'auto_optimize': True,
                'save_all_models': False,
                'models_dir': './fine_tuned_models',
                'use_lora': True,
                'use_quantization': False
            },
            
            # Formatos y modelos soportados
            'supported_formats': ['.txt', '.csv', '.json', '.pdf', '.db', '.jpg', '.png', '.docx', '.html'],
            'default_models': ['transformer', 'cnn', 'lstm', 'rnn', 'vae', 'ddpm', 'wgan_gp'],
            'advanced_models': {
                'generative': ['vae', 'beta_vae', 'conditional_vae', 'ddpm', 'conditional_ddpm'],
                'adversarial': ['wgan_gp', 'conditional_gan', 'cycle_gan', 'stylegan2'],
                'transformers': ['bert', 'gpt2', 't5', 'roberta', 'custom', 'hybrid']
            },
            
            # Configuración de inteligencia autónoma
            'intelligence_config': {
                'auto_model_selection': True,
                'adaptive_architecture': True,
                'continuous_optimization': True,
                'self_improvement': True,
                'knowledge_retention': True
            }
        }
    
    def discover_data_sources(self, base_path: str = None) -> List[str]:
        """Descubre automáticamente fuentes de datos disponibles"""
        base_path = base_path or self.config['data_sources_path']
        data_sources = []
        
        if not os.path.exists(base_path):
            logger.warning(f"Ruta de datos no encontrada: {base_path}")
            return data_sources
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in self.config['supported_formats']:
                    data_sources.append(file_path)
                    logger.info(f"Fuente de datos descubierta: {file_path}")
        
        return data_sources
    
    def extract_and_analyze_data(self, data_sources: List[str]) -> Dict[str, Any]:
        """Extrae y analiza datos de múltiples fuentes"""
        analysis_results = {
            'total_sources': len(data_sources),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'data_summary': {},
            'extraction_details': []
        }
        
        for source in data_sources:
            try:
                # Extraer datos
                extracted_data = self.data_extractor.extract_data(source)
                
                # Analizar datos extraídos
                analysis = self.data_extractor.analyze_extracted_data(extracted_data)
                
                analysis_results['successful_extractions'] += 1
                analysis_results['extraction_details'].append({
                    'source': source,
                    'status': 'success',
                    'analysis': analysis
                })
                
                # Actualizar resumen
                data_type = analysis.get('data_type', 'unknown')
                if data_type not in analysis_results['data_summary']:
                    analysis_results['data_summary'][data_type] = 0
                analysis_results['data_summary'][data_type] += 1
                
                logger.info(f"Datos extraídos y analizados exitosamente: {source}")
                
            except Exception as e:
                analysis_results['failed_extractions'] += 1
                analysis_results['extraction_details'].append({
                    'source': source,
                    'status': 'error',
                    'error': str(e)
                })
                logger.error(f"Error extrayendo datos de {source}: {str(e)}")
        
        return analysis_results
    
    def generate_intelligent_prompts(self, data_analysis: Dict[str, Any], 
                                   objectives: List[str] = None) -> List[Dict[str, Any]]:
        """Genera prompts inteligentes basados en el análisis de datos"""
        objectives = objectives or [
            "clasificación de texto",
            "generación de contenido",
            "análisis de sentimientos",
            "extracción de entidades",
            "resumen automático"
        ]
        
        generated_prompts = []
        
        for detail in data_analysis['extraction_details']:
            if detail['status'] == 'success':
                source = detail['source']
                analysis = detail['analysis']
                
                for objective in objectives:
                    try:
                        # Generar prompt optimizado
                        prompt_data = self.prompt_generator.create_optimized_prompt(
                            task_type=objective,
                            context=analysis.get('sample_content', ''),
                            examples=[],
                            optimization_level='advanced'
                        )
                        
                        generated_prompts.append({
                            'source': source,
                            'objective': objective,
                            'prompt': prompt_data,
                            'data_type': analysis.get('data_type'),
                            'complexity': analysis.get('complexity_score', 0.5)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error generando prompt para {objective}: {str(e)}")
        
        logger.info(f"Generados {len(generated_prompts)} prompts inteligentes")
        return generated_prompts
    
    def train_autonomous_ecosystem(self, data_sources: List[str], 
                                 cycles: int = 3) -> Dict[str, Any]:
        """Entrena un ecosistema autónomo de modelos"""
        logger.info(f"Iniciando entrenamiento autónomo con {len(data_sources)} fuentes de datos")
        
        # Configurar modelos por defecto
        model_configs = []
        for model_type in self.config['default_models']:
            model_configs.append({
                'type': model_type,
                'name': f'{model_type}_autonomous',
                'auto_optimize': True
            })
        
        # Entrenar usando el ecosistema avanzado
        training_results = self.ecosystem_trainer.autonomous_training_cycle(
            data_sources, 
            cycles=cycles
        )
        
        # Entrenar usando el ecosistema autónomo
        autonomous_results = self.autonomous_ecosystem.autonomous_training_cycle(
            data_sources=data_sources,
            cycles=cycles,
            auto_optimize=True
        )
        
        return {
            'ecosystem_training': training_results,
            'autonomous_training': autonomous_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def interactive_model_testing(self, prompts: List[str], 
                                tasks: List[str] = None) -> Dict[str, Any]:
        """Prueba interactiva de modelos con múltiples prompts"""
        tasks = tasks or ['text_generation', 'text_classification', 'question_answering']
        
        testing_results = {
            'total_prompts': len(prompts),
            'total_tasks': len(tasks),
            'results': [],
            'performance_summary': {}
        }
        
        for i, prompt in enumerate(prompts):
            prompt_results = {}
            
            for task in tasks:
                try:
                    # Probar con cada tarea
                    response = self.model_interaction.interact_with_model(prompt, task)
                    
                    prompt_results[task] = {
                        'response': response,
                        'status': 'success',
                        'response_length': len(response)
                    }
                    
                except Exception as e:
                    prompt_results[task] = {
                        'response': None,
                        'status': 'error',
                        'error': str(e)
                    }
            
            testing_results['results'].append({
                'prompt_index': i,
                'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'task_results': prompt_results
            })
            
            logger.info(f"Prompt {i+1}/{len(prompts)} procesado")
        
        # Generar resumen de rendimiento
        for task in tasks:
            successful_responses = sum(
                1 for result in testing_results['results'] 
                if result['task_results'].get(task, {}).get('status') == 'success'
            )
            
            testing_results['performance_summary'][task] = {
                'success_rate': successful_responses / len(prompts),
                'total_attempts': len(prompts)
            }
        
        return testing_results
    
    def generate_comprehensive_report(self, data_analysis: Dict[str, Any], 
                                    training_results: Dict[str, Any],
                                    testing_results: Dict[str, Any] = None) -> str:
        """Genera un reporte comprehensivo del entrenamiento"""
        report_lines = [
            "=" * 80,
            "REPORTE COMPREHENSIVO - BOT ENTRENADOR GENERAL",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. ANÁLISIS DE DATOS",
            "-" * 40,
            f"Total de fuentes procesadas: {data_analysis['total_sources']}",
            f"Extracciones exitosas: {data_analysis['successful_extractions']}",
            f"Extracciones fallidas: {data_analysis['failed_extractions']}",
            "",
            "Resumen por tipo de datos:",
        ]
        
        for data_type, count in data_analysis['data_summary'].items():
            report_lines.append(f"  - {data_type}: {count} fuentes")
        
        report_lines.extend([
            "",
            "2. RESULTADOS DE ENTRENAMIENTO",
            "-" * 40
        ])
        
        if 'ecosystem_training' in training_results:
            ecosystem_results = training_results['ecosystem_training']
            report_lines.append(f"Ciclos de entrenamiento completados: {ecosystem_results.get('cycles_completed', 0)}")
            
            if 'final_ecosystem_status' in ecosystem_results:
                status = ecosystem_results['final_ecosystem_status']
                report_lines.append(f"Modelos registrados: {len(status.get('registered_models', []))}")
        
        if testing_results:
            report_lines.extend([
                "",
                "3. RESULTADOS DE PRUEBAS",
                "-" * 40,
                f"Prompts probados: {testing_results['total_prompts']}",
                f"Tareas evaluadas: {testing_results['total_tasks']}",
                "",
                "Tasas de éxito por tarea:"
            ])
            
            for task, summary in testing_results['performance_summary'].items():
                success_rate = summary['success_rate'] * 100
                report_lines.append(f"  - {task}: {success_rate:.1f}%")
        
        report_lines.extend([
            "",
            "4. RECOMENDACIONES",
            "-" * 40,
            "- Continuar con ciclos de entrenamiento autónomo",
            "- Monitorear métricas de rendimiento regularmente",
            "- Expandir fuentes de datos para mejor generalización",
            "- Implementar validación cruzada para modelos críticos",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_complete_pipeline(self, base_data_path: str = None, 
                            training_cycles: int = 3) -> Dict[str, Any]:
        """Ejecuta el pipeline completo del bot entrenador"""
        logger.info("Iniciando pipeline completo del Bot Entrenador General")
        
        try:
            # 1. Descubrir fuentes de datos
            data_sources = self.discover_data_sources(base_data_path)
            
            if not data_sources:
                logger.warning("No se encontraron fuentes de datos")
                return {'status': 'error', 'message': 'No se encontraron fuentes de datos'}
            
            # 2. Extraer y analizar datos
            data_analysis = self.extract_and_analyze_data(data_sources)
            
            # 3. Generar prompts inteligentes
            intelligent_prompts = self.generate_intelligent_prompts(data_analysis)
            
            # 4. Entrenar ecosistema autónomo
            training_results = self.train_autonomous_ecosystem(data_sources, training_cycles)
            
            # 5. Probar modelos interactivamente
            test_prompts = [prompt['prompt'] for prompt in intelligent_prompts[:10]]  # Primeros 10
            testing_results = self.interactive_model_testing(test_prompts)
            
            # 6. Generar reporte
            comprehensive_report = self.generate_comprehensive_report(
                data_analysis, training_results, testing_results
            )
            
            # 7. Guardar resultados
            output_path = self.config['output_path']
            os.makedirs(output_path, exist_ok=True)
            
            report_file = os.path.join(output_path, f'bot_entrenador_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(comprehensive_report)
            
            logger.info(f"Pipeline completado. Reporte guardado en: {report_file}")
            
            return {
                'status': 'success',
                'data_analysis': data_analysis,
                'intelligent_prompts': len(intelligent_prompts),
                'training_results': training_results,
                'testing_results': testing_results,
                'report_file': report_file,
                'comprehensive_report': comprehensive_report
            }
            
        except Exception as e:
            logger.error(f"Error en pipeline completo: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Función principal mejorada"""
    print("🤖 Bot Entrenador General - Sistema Autónomo de IA")
    print("=" * 60)
    
    try:
        # Inicializar bot
        bot = BotEntrenadorGeneral()
        
        # Ejecutar pipeline completo
        results = bot.run_complete_pipeline()
        
        if results['status'] == 'success':
            print("\n✅ Pipeline ejecutado exitosamente!")
            print(f"📊 Fuentes de datos analizadas: {results['data_analysis']['total_sources']}")
            print(f"🧠 Prompts inteligentes generados: {results['intelligent_prompts']}")
            print(f"📄 Reporte guardado en: {results['report_file']}")
            
            # Mostrar reporte en consola
            print("\n" + "="*60)
            print("REPORTE RESUMIDO:")
            print("="*60)
            print(results['comprehensive_report'])
            
        else:
            print(f"❌ Error en la ejecución: {results.get('error', 'Error desconocido')}")
    
    except Exception as e:
        logger.error(f"Error crítico en main: {str(e)}")
        print(f"❌ Error crítico: {str(e)}")

    def _init_system_database(self):
        """Inicializar base de datos del sistema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de tareas del sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                task_type TEXT,
                status TEXT,
                priority INTEGER,
                config TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error_message TEXT
            )
        ''')
        
        # Tabla de métricas de rendimiento
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_type TEXT,
                metric_name TEXT,
                metric_value REAL,
                context TEXT
            )
        ''')
        
        # Tabla de modelos entrenados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trained_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT UNIQUE,
                model_type TEXT,
                architecture TEXT,
                training_data_hash TEXT,
                performance_score REAL,
                model_path TEXT,
                created_at TEXT,
                metadata TEXT
            )
        ''')
        
        # Tabla de conocimiento del sistema RAG
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content_hash TEXT,
                embedding_vector BLOB,
                metadata TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Base de datos del sistema inicializada")

    def start_autonomous_processing(self):
        """Iniciar procesamiento autónomo de tareas"""
        max_threads = self.config.get('max_concurrent_tasks', 4)
        
        for i in range(max_threads):
            thread = threading.Thread(
                target=self._autonomous_task_processor,
                name=f"TaskProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Procesamiento autónomo iniciado con {max_threads} hilos")

    def _autonomous_task_processor(self):
        """Procesador autónomo de tareas"""
        while not self.stop_processing.is_set():
            try:
                # Obtener tarea de la cola (timeout de 5 segundos)
                priority, task = self.task_queue.get(timeout=5)
                
                # Procesar tarea
                self._execute_autonomous_task(task)
                
                # Marcar tarea como completada
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en procesador autónomo: {e}")
                time.sleep(1)

    def _execute_autonomous_task(self, task: Dict[str, Any]):
        """Ejecutar tarea autónoma"""
        task_id = task['task_id']
        task_type = task['task_type']
        
        try:
            logger.info(f"Ejecutando tarea autónoma: {task_id} ({task_type})")
            
            # Actualizar estado en base de datos
            self._update_task_status(task_id, 'running', started_at=datetime.now().isoformat())
            
            # Ejecutar según tipo de tarea
            if task_type == 'data_discovery':
                result = self._autonomous_data_discovery(task)
            elif task_type == 'model_training':
                result = self._autonomous_model_training(task)
            elif task_type == 'fine_tuning':
                result = self._autonomous_fine_tuning(task)
            elif task_type == 'rag_query':
                result = self._autonomous_rag_query(task)
            elif task_type == 'knowledge_update':
                result = self._autonomous_knowledge_update(task)
            elif task_type == 'performance_optimization':
                result = self._autonomous_performance_optimization(task)
            else:
                result = self._execute_custom_task(task)
            
            # Actualizar estado como completado
            self._update_task_status(
                task_id, 'completed', 
                completed_at=datetime.now().isoformat(),
                result=json.dumps(result)
            )
            
            # Actualizar métricas del sistema
            self._update_system_metrics(task_type, result)
            
            logger.info(f"Tarea completada exitosamente: {task_id}")
            
        except Exception as e:
            logger.error(f"Error ejecutando tarea {task_id}: {e}")
            self._update_task_status(
                task_id, 'failed',
                completed_at=datetime.now().isoformat(),
                error_message=str(e)
            )

    def _autonomous_data_discovery(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Descubrimiento autónomo de datos"""
        config = task.get('config', {})
        base_path = config.get('base_path', self.config['data_sources_path'])
        
        # Descubrir fuentes de datos
        data_sources = self.discover_data_sources(base_path)
        
        # Analizar y procesar datos
        analysis_results = self.extract_and_analyze_data(data_sources)
        
        # Actualizar sistema RAG con nuevos datos
        for source in data_sources:
            self.rag_system.add_data_source(source)
        
        return {
            'discovered_sources': len(data_sources),
            'analysis_results': analysis_results,
            'rag_updated': True
        }

    def _autonomous_model_training(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Entrenamiento autónomo de modelos"""
        config = task.get('config', {})
        
        # Determinar tipo de modelo basado en datos
        data_characteristics = config.get('data_characteristics', {})
        model_type = self._select_optimal_model_type(data_characteristics)
        
        # Crear modelo adaptativo
        model = create_adaptive_model(model_type, data_characteristics)
        
        # Entrenar usando el ecosistema
        training_results = self.autonomous_ecosystem.autonomous_training_cycle(
            data_sources=config.get('data_sources', []),
            cycles=config.get('cycles', 3),
            auto_optimize=True
        )
        
        return {
            'model_type': model_type,
            'training_results': training_results,
            'model_created': True
        }

    def _autonomous_fine_tuning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tuning autónomo"""
        config = task.get('config', {})
        
        # Agregar trabajo de fine-tuning al sistema
        job_id = self.fine_tuning_system.add_training_job(config)
        
        return {
            'fine_tuning_job_id': job_id,
            'status': 'queued'
        }

    def _autonomous_rag_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Consulta autónoma RAG"""
        config = task.get('config', {})
        query = config.get('query', '')
        
        # Realizar consulta RAG
        result = self.rag_system.query(query)
        
        # Actualizar métricas
        self.system_state['performance_metrics']['total_queries_answered'] += 1
        
        return result

    def _autonomous_knowledge_update(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Actualización autónoma de conocimiento"""
        # Actualizar base de conocimiento RAG
        self.rag_system.update_knowledge_base()
        
        # Obtener reporte de rendimiento
        performance_report = self.rag_system.get_performance_report()
        
        return {
            'knowledge_updated': True,
            'performance_report': performance_report
        }

    def _autonomous_performance_optimization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimización autónoma de rendimiento"""
        config = task.get('config', {})
        
        # Analizar métricas actuales
        current_metrics = self._get_current_performance_metrics()
        
        # Identificar áreas de mejora
        optimization_suggestions = self._analyze_performance_bottlenecks(current_metrics)
        
        # Aplicar optimizaciones automáticas
        applied_optimizations = []
        for suggestion in optimization_suggestions:
            if suggestion['auto_applicable']:
                self._apply_optimization(suggestion)
                applied_optimizations.append(suggestion['type'])
        
        return {
            'current_metrics': current_metrics,
            'optimization_suggestions': optimization_suggestions,
            'applied_optimizations': applied_optimizations
        }

    def _select_optimal_model_type(self, data_characteristics: Dict[str, Any]) -> str:
        """Seleccionar tipo de modelo óptimo basado en características de datos"""
        data_type = data_characteristics.get('data_type', 'unknown')
        complexity = data_characteristics.get('complexity_score', 0.5)
        task_type = data_characteristics.get('task_type', 'classification')
        
        # Lógica de selección inteligente
        if data_type == 'text':
            if task_type in ['generation', 'completion']:
                return 'transformer' if complexity > 0.7 else 'lstm'
            else:
                return 'bert' if complexity > 0.5 else 'transformer'
        
        elif data_type == 'image':
            if task_type == 'generation':
                return 'ddpm' if complexity > 0.7 else 'wgan_gp'
            else:
                return 'cnn'
        
        elif data_type == 'sequence':
            return 'lstm' if complexity > 0.5 else 'rnn'
        
        elif data_type == 'tabular':
            if task_type == 'generation':
                return 'vae'
            else:
                return 'transformer'
        
        else:
            # Modelo por defecto
            return 'transformer'

    def add_autonomous_task(self, task_type: str, config: Dict[str, Any], 
                           priority: int = 1) -> str:
        """Agregar tarea autónoma al sistema"""
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.system_state['active_tasks'])}"
        
        task = {
            'task_id': task_id,
            'task_type': task_type,
            'config': config,
            'priority': priority,
            'created_at': datetime.now().isoformat()
        }
        
        # Agregar a la cola (prioridad negativa para orden correcto)
        self.task_queue.put((-priority, task))
        
        # Registrar en base de datos
        self._register_task_in_db(task)
        
        # Actualizar estado del sistema
        self.system_state['active_tasks'][task_id] = task
        
        logger.info(f"Tarea autónoma agregada: {task_id} ({task_type})")
        
        return task_id

    def query_intelligent_system(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Consulta inteligente al sistema completo"""
        start_time = time.time()
        
        # Analizar la consulta para determinar el mejor enfoque
        query_analysis = self._analyze_query_intent(query)
        
        results = {}
        
        # Consulta RAG para información contextual
        if query_analysis.get('needs_context', True):
            rag_result = self.rag_system.query(query)
            results['rag_response'] = rag_result
        
        # Generar respuesta usando modelos avanzados
        if query_analysis.get('needs_generation', True):
            generation_result = self._generate_intelligent_response(query, context)
            results['generated_response'] = generation_result
        
        # Buscar modelos relevantes si es necesario
        if query_analysis.get('needs_models', False):
            model_suggestions = self._suggest_relevant_models(query, context)
            results['model_suggestions'] = model_suggestions
        
        # Crear respuesta final combinada
        final_response = self._combine_intelligent_responses(results, query)
        
        response_time = time.time() - start_time
        
        return {
            'query': query,
            'final_response': final_response,
            'detailed_results': results,
            'query_analysis': query_analysis,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }

    def get_system_intelligence_report(self) -> Dict[str, Any]:
        """Obtener reporte de inteligencia del sistema"""
        # Estado de componentes
        rag_status = self.rag_system.get_performance_report()
        ft_status = self.fine_tuning_system.get_system_status()
        ecosystem_status = self.autonomous_ecosystem.get_ecosystem_status()
        
        # Métricas del sistema
        system_metrics = self._get_current_performance_metrics()
        
        # Modelos disponibles
        available_models = get_available_models()
        
        # Tareas activas y completadas
        active_tasks = len(self.system_state['active_tasks'])
        completed_tasks = len(self.system_state['completed_tasks'])
        
        return {
            'system_overview': {
                'status': 'operational',
                'uptime': self._calculate_uptime(),
                'active_tasks': active_tasks,
                'completed_tasks': completed_tasks,
                'total_models_available': len(available_models)
            },
            'component_status': {
                'rag_system': rag_status,
                'fine_tuning_system': ft_status,
                'autonomous_ecosystem': ecosystem_status
            },
            'performance_metrics': system_metrics,
            'available_models': {
                'total': len(available_models),
                'by_category': AVAILABLE_MODELS
            },
            'intelligence_capabilities': {
                'autonomous_learning': True,
                'adaptive_architecture': True,
                'continuous_optimization': True,
                'knowledge_retention': True,
                'multi_modal_processing': True
            },
            'timestamp': datetime.now().isoformat()
        }

    # Métodos auxiliares para completar la funcionalidad
    def _update_task_status(self, task_id: str, status: str, **kwargs):
        """Actualizar estado de tarea en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = ['status = ?']
        values = [status]
        
        for field, value in kwargs.items():
            update_fields.append(f'{field} = ?')
            values.append(value)
        
        values.append(task_id)
        
        cursor.execute(f'''
            UPDATE system_tasks 
            SET {', '.join(update_fields)}
            WHERE task_id = ?
        ''', values)
        
        conn.commit()
        conn.close()
    
    def _register_task_in_db(self, task: Dict[str, Any]):
        """Registrar tarea en base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_tasks (task_id, task_type, status, priority, config, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            task['task_id'],
            task['task_type'],
            'queued',
            task['priority'],
            json.dumps(task['config']),
            task['created_at']
        ))
        
        conn.commit()
        conn.close()
    
    def _update_system_metrics(self, task_type: str, result: Dict[str, Any]):
        """Actualizar métricas del sistema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Registrar métricas específicas según el tipo de tarea
        if task_type == 'model_training':
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_type, metric_name, metric_value, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, 'training', 'models_trained', 1, json.dumps(result)))
            
            self.system_state['performance_metrics']['total_models_trained'] += 1
        
        elif task_type == 'rag_query':
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_type, metric_name, metric_value, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, 'rag', 'queries_answered', 1, json.dumps(result)))
        
        conn.commit()
        conn.close()
    
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales de rendimiento"""
        return self.system_state['performance_metrics'].copy()
    
    def _analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analizar cuellos de botella de rendimiento"""
        suggestions = []
        
        # Analizar tiempo de respuesta
        if metrics.get('average_training_time', 0) > 300:  # 5 minutos
            suggestions.append({
                'type': 'training_optimization',
                'description': 'Tiempo de entrenamiento alto, considerar optimización',
                'auto_applicable': True,
                'priority': 2
            })
        
        # Analizar tasa de éxito
        if metrics.get('success_rate', 1.0) < 0.8:
            suggestions.append({
                'type': 'error_reduction',
                'description': 'Tasa de éxito baja, revisar configuraciones',
                'auto_applicable': False,
                'priority': 1
            })
        
        return suggestions
    
    def _apply_optimization(self, suggestion: Dict[str, Any]):
        """Aplicar optimización automática"""
        opt_type = suggestion['type']
        
        if opt_type == 'training_optimization':
            # Reducir batch size para acelerar entrenamiento
            if self.config['training_config']['batch_size'] > 16:
                self.config['training_config']['batch_size'] //= 2
                logger.info("Optimización aplicada: Reducido batch size")
    
    def _calculate_uptime(self) -> str:
        """Calcular tiempo de actividad del sistema"""
        init_time = datetime.fromisoformat(self.system_state['initialized_at'])
        uptime = datetime.now() - init_time
        return str(uptime)
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analizar intención de la consulta"""
        query_lower = query.lower()
        
        # Análisis simple de intención
        needs_context = any(word in query_lower for word in ['qué', 'cómo', 'cuál', 'información', 'datos'])
        needs_generation = any(word in query_lower for word in ['genera', 'crea', 'escribe', 'produce'])
        needs_models = any(word in query_lower for word in ['modelo', 'entrenar', 'algoritmo', 'red neuronal'])
        
        return {
            'needs_context': needs_context,
            'needs_generation': needs_generation,
            'needs_models': needs_models,
            'complexity': len(query.split()) / 20.0  # Complejidad basada en longitud
        }
    
    def _generate_intelligent_response(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generar respuesta inteligente"""
        # Usar el sistema de interacción de modelos
        return self.model_interaction.interact_with_model(query, 'text_generation')
    
    def _suggest_relevant_models(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Sugerir modelos relevantes para la consulta"""
        query_lower = query.lower()
        suggestions = []
        
        if any(word in query_lower for word in ['imagen', 'foto', 'visual']):
            suggestions.extend(['cnn', 'ddpm', 'wgan_gp'])
        
        if any(word in query_lower for word in ['texto', 'lenguaje', 'nlp']):
            suggestions.extend(['transformer', 'bert', 'gpt2'])
        
        if any(word in query_lower for word in ['secuencia', 'tiempo', 'serie']):
            suggestions.extend(['lstm', 'rnn'])
        
        if any(word in query_lower for word in ['generar', 'crear', 'sintetizar']):
            suggestions.extend(['vae', 'ddpm', 'wgan_gp'])
        
        return list(set(suggestions))  # Eliminar duplicados
    
    def _combine_intelligent_responses(self, results: Dict[str, Any], query: str) -> str:
        """Combinar respuestas inteligentes"""
        response_parts = []
        
        # Agregar respuesta RAG si existe
        if 'rag_response' in results:
            rag_resp = results['rag_response'].get('response', '')
            if rag_resp:
                response_parts.append(f"Información contextual: {rag_resp}")
        
        # Agregar respuesta generada
        if 'generated_response' in results:
            gen_resp = results['generated_response']
            if gen_resp:
                response_parts.append(f"Respuesta generada: {gen_resp}")
        
        # Agregar sugerencias de modelos
        if 'model_suggestions' in results and results['model_suggestions']:
            models = ', '.join(results['model_suggestions'])
            response_parts.append(f"Modelos recomendados: {models}")
        
        return "\n\n".join(response_parts) if response_parts else "No se pudo generar una respuesta adecuada."
    
    def _execute_custom_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea personalizada"""
        return {
            'status': 'completed',
            'message': 'Tarea personalizada ejecutada',
            'task_id': task['task_id']
        }

def main():
    """Función principal del Bot Entrenador General Avanzado"""
    print("🤖 Bot Entrenador General - Sistema Autónomo Avanzado de IA")
    print("=" * 70)
    print("🚀 Integrando RAG, Fine-Tuning y Modelos Avanzados")
    print("=" * 70)
    
    try:
        # Inicializar bot avanzado
        bot = BotEntrenadorGeneral()
        
        # Mostrar capacidades del sistema
        print("\n🧠 Capacidades del Sistema:")
        print("   • Modelos Avanzados: Transformer, VAE, Diffusion, GAN")
        print("   • Sistema RAG Autónomo")
        print("   • Fine-Tuning Automático con LoRA")
        print("   • Procesamiento Concurrente")
        print("   • Optimización Continua")
        
        # Agregar algunas tareas autónomas de ejemplo
        print("\n📋 Agregando tareas autónomas...")
        
        # Tarea de descubrimiento de datos
        discovery_task = bot.add_autonomous_task(
            'data_discovery',
            {'base_path': './data'},
            priority=3
        )
        print(f"   ✓ Tarea de descubrimiento: {discovery_task}")
        
        # Tarea de actualización de conocimiento
        knowledge_task = bot.add_autonomous_task(
            'knowledge_update',
            {},
            priority=2
        )
        print(f"   ✓ Tarea de actualización: {knowledge_task}")
        
        # Consulta inteligente de ejemplo
        print("\n🔍 Realizando consulta inteligente...")
        query_result = bot.query_intelligent_system(
            "¿Qué modelos son mejores para generar imágenes?"
        )
        print(f"   Respuesta: {query_result['final_response'][:200]}...")
        
        # Obtener reporte de inteligencia
        print("\n📊 Generando reporte de inteligencia del sistema...")
        intelligence_report = bot.get_system_intelligence_report()
        
        print(f"\n✅ Sistema Operacional:")
        print(f"   • Estado: {intelligence_report['system_overview']['status']}")
        print(f"   • Tareas Activas: {intelligence_report['system_overview']['active_tasks']}")
        print(f"   • Modelos Disponibles: {intelligence_report['system_overview']['total_models_available']}")
        print(f"   • Tiempo de Actividad: {intelligence_report['system_overview']['uptime']}")
        
        print(f"\n🎯 Capacidades de IA:")
        capabilities = intelligence_report['intelligence_capabilities']
        for capability, enabled in capabilities.items():
            status = "✓" if enabled else "✗"
            print(f"   {status} {capability.replace('_', ' ').title()}")
        
        # Ejecutar pipeline completo original
        print("\n🔄 Ejecutando pipeline completo...")
        results = bot.run_complete_pipeline()
        
        if results['status'] == 'success':
            print("\n🎉 ¡Pipeline ejecutado exitosamente!")
            print(f"📊 Fuentes de datos analizadas: {results['data_analysis']['total_sources']}")
            print(f"🧠 Prompts inteligentes generados: {results['intelligent_prompts']}")
            print(f"📄 Reporte guardado en: {results['report_file']}")
        else:
            print(f"❌ Error en la ejecución: {results.get('error', 'Error desconocido')}")
        
        print("\n🌟 Sistema Bot Entrenador General Avanzado listo para uso autónomo!")
        print("   El sistema continuará procesando tareas en segundo plano...")
        
    except Exception as e:
        logger.error(f"Error crítico en main: {str(e)}")
        print(f"❌ Error crítico: {str(e)}")

if __name__ == "__main__":
    main()
