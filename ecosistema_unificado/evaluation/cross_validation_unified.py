"""
Sistema de Validación Cruzada y Métricas Avanzadas Unificado
============================================================

Sistema robusto para evaluación de modelos con validación cruzada,
métricas avanzadas y análisis estadístico completo.

Autor: Sistema de IA Avanzado
Versión: 1.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
from datetime import datetime
import pandas as pd

class CrossValidationUnified:
    """Sistema unificado de validación cruzada y métricas avanzadas"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Inicializa el sistema de validación cruzada
        
        Args:
            n_splits: Número de divisiones para validación cruzada
            random_state: Semilla para reproducibilidad
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.cv_results = {}
        self.detailed_metrics = {}
        
    def k_fold_cross_validation(self,
                               model_builder_func,
                               x_data: np.ndarray,
                               y_data: np.ndarray,
                               model_params: Dict[str, Any] = None,
                               stratified: bool = True) -> Dict[str, Any]:
        """
        Realiza validación cruzada K-Fold
        
        Args:
            model_builder_func: Función que construye el modelo
            x_data: Datos de entrada
            y_data: Etiquetas
            model_params: Parámetros del modelo
            stratified: Si usar StratifiedKFold
            
        Returns:
            Resultados de validación cruzada
        """
        if model_params is None:
            model_params = {}
            
        self.logger.info(f"Iniciando validación cruzada {self.n_splits}-fold")
        
        # Seleccionar tipo de validación cruzada
        if stratified and len(np.unique(y_data)) > 1:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            self.logger.info("Usando StratifiedKFold")
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            self.logger.info("Usando KFold")
        
        # Métricas para cada fold
        fold_results = []
        fold_predictions = []
        fold_true_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(x_data, y_data)):
            self.logger.info(f"Procesando fold {fold + 1}/{self.n_splits}")
            
            # Dividir datos
            x_train_fold = x_data[train_idx]
            y_train_fold = y_data[train_idx]
            x_val_fold = x_data[val_idx]
            y_val_fold = y_data[val_idx]
            
            try:
                # Construir modelo
                model = model_builder_func(**model_params)
                
                # Entrenar modelo
                history = model.train(
                    x_train_fold, y_train_fold,
                    x_val_fold, y_val_fold,
                    epochs=model_params.get('epochs', 50),
                    batch_size=model_params.get('batch_size', 32),
                    verbose=0
                )
                
                # Evaluar modelo
                results = model.evaluate(x_val_fold, y_val_fold)
                
                # Obtener predicciones
                predictions = model.predict(x_val_fold)
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Clasificación multiclase
                    y_pred = np.argmax(predictions, axis=1)
                    y_pred_proba = predictions
                else:
                    # Clasificación binaria
                    y_pred = (predictions > 0.5).astype(int).flatten()
                    y_pred_proba = predictions.flatten()
                
                # Calcular métricas detalladas
                fold_metrics = self._calculate_detailed_metrics(
                    y_val_fold, y_pred, y_pred_proba
                )
                
                fold_metrics.update({
                    'fold': fold + 1,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'model_accuracy': results.get('accuracy', 0),
                    'model_loss': results.get('loss', 0)
                })
                
                fold_results.append(fold_metrics)
                fold_predictions.extend(y_pred)
                fold_true_labels.extend(y_val_fold)
                
                self.logger.info(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error en fold {fold + 1}: {e}")
                continue
        
        # Calcular estadísticas agregadas
        cv_summary = self._calculate_cv_summary(fold_results)
        
        # Métricas globales con todas las predicciones
        global_metrics = self._calculate_detailed_metrics(
            np.array(fold_true_labels),
            np.array(fold_predictions),
            np.array(fold_predictions)  # Simplificado para este ejemplo
        )
        
        self.cv_results = {
            'fold_results': fold_results,
            'cv_summary': cv_summary,
            'global_metrics': global_metrics,
            'n_splits': self.n_splits,
            'total_samples': len(x_data),
            'stratified': stratified
        }
        
        self.logger.info(f"Validación cruzada completada")
        self.logger.info(f"Accuracy promedio: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
        
        return self.cv_results
    
    def _calculate_detailed_metrics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calcula métricas detalladas"""
        
        metrics = {}
        
        try:
            # Métricas básicas
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Métricas por clase
            if len(np.unique(y_true)) > 2:
                # Multiclase
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            else:
                # Binaria
                metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
                metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
                metrics['f1_score_binary'] = f1_score(y_true, y_pred, zero_division=0)
                
                # AUC-ROC para clasificación binaria
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                    except:
                        metrics['auc_roc'] = 0.0
            
            # Matriz de confusión (como lista para serialización JSON)
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Métricas adicionales
            metrics['true_positives'] = np.sum((y_true == 1) & (y_pred == 1))
            metrics['true_negatives'] = np.sum((y_true == 0) & (y_pred == 0))
            metrics['false_positives'] = np.sum((y_true == 0) & (y_pred == 1))
            metrics['false_negatives'] = np.sum((y_true == 1) & (y_pred == 0))
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas: {e}")
            # Métricas por defecto en caso de error
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        return metrics
    
    def _calculate_cv_summary(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Calcula estadísticas resumidas de validación cruzada"""
        
        if not fold_results:
            return {}
        
        # Extraer métricas numéricas
        numeric_metrics = {}
        for key in fold_results[0].keys():
            if isinstance(fold_results[0][key], (int, float)) and key != 'fold':
                values = [fold[key] for fold in fold_results if key in fold]
                if values:
                    numeric_metrics[f'mean_{key}'] = np.mean(values)
                    numeric_metrics[f'std_{key}'] = np.std(values)
                    numeric_metrics[f'min_{key}'] = np.min(values)
                    numeric_metrics[f'max_{key}'] = np.max(values)
        
        return numeric_metrics
    
    def bootstrap_evaluation(self,
                           model,
                           x_test: np.ndarray,
                           y_test: np.ndarray,
                           n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Evaluación con bootstrap para intervalos de confianza
        
        Args:
            model: Modelo entrenado
            x_test: Datos de prueba
            y_test: Etiquetas de prueba
            n_bootstrap: Número de muestras bootstrap
            
        Returns:
            Estadísticas con intervalos de confianza
        """
        self.logger.info(f"Iniciando evaluación bootstrap con {n_bootstrap} muestras")
        
        bootstrap_scores = []
        n_samples = len(x_test)
        
        for i in range(n_bootstrap):
            # Muestreo con reemplazo
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            x_boot = x_test[indices]
            y_boot = y_test[indices]
            
            try:
                # Evaluar modelo
                results = model.evaluate(x_boot, y_boot)
                bootstrap_scores.append(results.get('accuracy', 0))
                
            except Exception as e:
                self.logger.warning(f"Error en bootstrap {i+1}: {e}")
                continue
        
        # Calcular estadísticas
        bootstrap_scores = np.array(bootstrap_scores)
        
        bootstrap_stats = {
            'mean_accuracy': np.mean(bootstrap_scores),
            'std_accuracy': np.std(bootstrap_scores),
            'confidence_interval_95': {
                'lower': np.percentile(bootstrap_scores, 2.5),
                'upper': np.percentile(bootstrap_scores, 97.5)
            },
            'confidence_interval_99': {
                'lower': np.percentile(bootstrap_scores, 0.5),
                'upper': np.percentile(bootstrap_scores, 99.5)
            },
            'n_bootstrap': len(bootstrap_scores),
            'min_accuracy': np.min(bootstrap_scores),
            'max_accuracy': np.max(bootstrap_scores)
        }
        
        self.logger.info(f"Bootstrap completado: {bootstrap_stats['mean_accuracy']:.4f} ± {bootstrap_stats['std_accuracy']:.4f}")
        
        return bootstrap_stats
    
    def comprehensive_evaluation(self,
                               model_builder_func,
                               x_data: np.ndarray,
                               y_data: np.ndarray,
                               x_test: np.ndarray = None,
                               y_test: np.ndarray = None,
                               model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluación comprehensiva combinando validación cruzada y bootstrap
        
        Args:
            model_builder_func: Función constructora del modelo
            x_data: Datos de entrenamiento/validación
            y_data: Etiquetas de entrenamiento/validación
            x_test: Datos de prueba (opcional)
            y_test: Etiquetas de prueba (opcional)
            model_params: Parámetros del modelo
            
        Returns:
            Evaluación completa
        """
        self.logger.info("Iniciando evaluación comprehensiva")
        
        comprehensive_results = {}
        
        # 1. Validación cruzada
        cv_results = self.k_fold_cross_validation(
            model_builder_func, x_data, y_data, model_params
        )
        comprehensive_results['cross_validation'] = cv_results
        
        # 2. Evaluación en conjunto de prueba (si está disponible)
        if x_test is not None and y_test is not None:
            self.logger.info("Evaluando en conjunto de prueba")
            
            # Entrenar modelo final con todos los datos
            final_model = model_builder_func(**(model_params or {}))
            final_model.train(
                x_data, y_data,
                epochs=model_params.get('epochs', 50) if model_params else 50,
                batch_size=model_params.get('batch_size', 32) if model_params else 32,
                verbose=0
            )
            
            # Evaluación estándar
            test_results = final_model.evaluate(x_test, y_test)
            
            # Bootstrap en conjunto de prueba
            bootstrap_results = self.bootstrap_evaluation(final_model, x_test, y_test)
            
            comprehensive_results['test_evaluation'] = {
                'standard_metrics': test_results,
                'bootstrap_statistics': bootstrap_results
            }
        
        # 3. Resumen final
        comprehensive_results['summary'] = self._generate_evaluation_summary(comprehensive_results)
        comprehensive_results['timestamp'] = datetime.now().isoformat()
        
        self.detailed_metrics = comprehensive_results
        
        self.logger.info("Evaluación comprehensiva completada")
        return comprehensive_results
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera resumen de la evaluación"""
        
        summary = {}
        
        # Resumen de validación cruzada
        if 'cross_validation' in results:
            cv = results['cross_validation']['cv_summary']
            summary['cv_mean_accuracy'] = cv.get('mean_accuracy', 0)
            summary['cv_std_accuracy'] = cv.get('std_accuracy', 0)
            summary['cv_confidence'] = f"{cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}"
        
        # Resumen de prueba
        if 'test_evaluation' in results:
            test = results['test_evaluation']
            summary['test_accuracy'] = test['standard_metrics'].get('accuracy', 0)
            
            if 'bootstrap_statistics' in test:
                boot = test['bootstrap_statistics']
                summary['test_confidence_95'] = boot['confidence_interval_95']
                summary['test_bootstrap_mean'] = boot['mean_accuracy']
        
        # Recomendaciones
        summary['recommendations'] = self._generate_recommendations(results)
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en los resultados"""
        
        recommendations = []
        
        if 'cross_validation' in results:
            cv_std = results['cross_validation']['cv_summary'].get('std_accuracy', 0)
            cv_mean = results['cross_validation']['cv_summary'].get('mean_accuracy', 0)
            
            if cv_std > 0.05:
                recommendations.append("Alta variabilidad entre folds - considerar más datos o regularización")
            
            if cv_mean < 0.8:
                recommendations.append("Accuracy baja - considerar ajustar hiperparámetros o arquitectura")
            elif cv_mean > 0.95:
                recommendations.append("Accuracy muy alta - verificar posible overfitting")
        
        if 'test_evaluation' in results and 'bootstrap_statistics' in results['test_evaluation']:
            boot_std = results['test_evaluation']['bootstrap_statistics']['std_accuracy']
            
            if boot_std > 0.03:
                recommendations.append("Alta incertidumbre en predicciones - considerar ensemble de modelos")
        
        if not recommendations:
            recommendations.append("Modelo muestra rendimiento estable y confiable")
        
        return recommendations
    
    def save_evaluation_results(self, filepath: str = 'evaluation_results.json'):
        """Guarda los resultados de evaluación"""
        
        if self.detailed_metrics:
            with open(filepath, 'w') as f:
                json.dump(self.detailed_metrics, f, indent=2, default=str)
            
            self.logger.info(f"Resultados guardados en {filepath}")
        else:
            self.logger.warning("No hay resultados para guardar")
    
    def generate_evaluation_report(self, save_path: str = 'evaluation_report.txt'):
        """Genera reporte detallado de evaluación"""
        
        if not self.detailed_metrics:
            self.logger.warning("No hay métricas para generar reporte")
            return
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("REPORTE DE EVALUACIÓN COMPREHENSIVA")
        report_lines.append("=" * 60)
        report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Validación cruzada
        if 'cross_validation' in self.detailed_metrics:
            cv = self.detailed_metrics['cross_validation']
            report_lines.append("VALIDACIÓN CRUZADA:")
            report_lines.append(f"  - Número de folds: {cv['n_splits']}")
            report_lines.append(f"  - Total de muestras: {cv['total_samples']}")
            report_lines.append(f"  - Accuracy promedio: {cv['cv_summary'].get('mean_accuracy', 0):.4f}")
            report_lines.append(f"  - Desviación estándar: {cv['cv_summary'].get('std_accuracy', 0):.4f}")
            report_lines.append("")
        
        # Evaluación en prueba
        if 'test_evaluation' in self.detailed_metrics:
            test = self.detailed_metrics['test_evaluation']
            report_lines.append("EVALUACIÓN EN CONJUNTO DE PRUEBA:")
            report_lines.append(f"  - Accuracy: {test['standard_metrics'].get('accuracy', 0):.4f}")
            
            if 'bootstrap_statistics' in test:
                boot = test['bootstrap_statistics']
                report_lines.append(f"  - Bootstrap mean: {boot['mean_accuracy']:.4f}")
                report_lines.append(f"  - Intervalo 95%: [{boot['confidence_interval_95']['lower']:.4f}, {boot['confidence_interval_95']['upper']:.4f}]")
            report_lines.append("")
        
        # Recomendaciones
        if 'summary' in self.detailed_metrics and 'recommendations' in self.detailed_metrics['summary']:
            report_lines.append("RECOMENDACIONES:")
            for rec in self.detailed_metrics['summary']['recommendations']:
                report_lines.append(f"  - {rec}")
        
        # Guardar reporte
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Reporte guardado en {save_path}")
        
        return '\n'.join(report_lines)