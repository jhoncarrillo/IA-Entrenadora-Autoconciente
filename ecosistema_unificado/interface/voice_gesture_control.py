#!/usr/bin/env python3
"""
Voice & Gesture Control System - Sistema de Control por Voz y Gestos
Control futurista mediante comandos de voz y gestos para la interfaz NeuroVision
"""

import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import time
import json
import re
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import queue
import pyaudio
import wave

class VoiceGestureController:
    """
    Sistema de control por voz y gestos para la interfaz NeuroVision
    """
    
    def __init__(self, callback_handler=None):
        # Inicializar sistemas
        self.callback_handler = callback_handler
        self.is_active = False
        self.voice_active = False
        self.gesture_active = False
        
        # Configuración de voz
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.setup_voice_engine()
        
        # Configuración de gestos
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Estados de gestos
        self.gesture_state = {
            'last_gesture': None,
            'gesture_confidence': 0.0,
            'gesture_start_time': None,
            'gesture_duration': 0.0,
            'hand_positions': [],
            'pose_landmarks': None,
            'face_landmarks': None,
            'eye_tracking': {'x': 0, 'y': 0},
            'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0}
        }
        
        # Comandos de voz
        self.voice_commands = {
            # Comandos de entrenamiento
            'entrenar': ['entrenar', 'train', 'iniciar entrenamiento', 'start training'],
            'parar': ['parar', 'stop', 'detener', 'halt', 'pause'],
            'continuar': ['continuar', 'continue', 'resume', 'reanudar'],
            
            # Comandos de visualización
            'mostrar': ['mostrar', 'show', 'display', 'visualizar'],
            'ocultar': ['ocultar', 'hide', 'esconder'],
            'zoom_in': ['acercar', 'zoom in', 'ampliar'],
            'zoom_out': ['alejar', 'zoom out', 'reducir'],
            'rotar': ['rotar', 'rotate', 'girar'],
            
            # Comandos de navegación
            'siguiente': ['siguiente', 'next', 'adelante'],
            'anterior': ['anterior', 'previous', 'atrás', 'back'],
            'inicio': ['inicio', 'home', 'principal'],
            'menu': ['menú', 'menu', 'opciones'],
            
            # Comandos de modelo
            'crear_modelo': ['crear modelo', 'create model', 'nuevo modelo'],
            'optimizar': ['optimizar', 'optimize', 'mejorar'],
            'evaluar': ['evaluar', 'evaluate', 'test'],
            
            # Comandos de datos
            'cargar_datos': ['cargar datos', 'load data', 'importar datos'],
            'analizar_datos': ['analizar datos', 'analyze data', 'explorar datos'],
            
            # Comandos de sistema
            'ayuda': ['ayuda', 'help', 'asistencia'],
            'configuracion': ['configuración', 'settings', 'config'],
            'guardar': ['guardar', 'save', 'salvar'],
            'cargar': ['cargar', 'load', 'abrir']
        }
        
        # Gestos reconocidos
        self.gesture_commands = {
            'point_up': 'scroll_up',
            'point_down': 'scroll_down',
            'point_left': 'navigate_left',
            'point_right': 'navigate_right',
            'thumbs_up': 'approve',
            'thumbs_down': 'disapprove',
            'open_palm': 'stop',
            'closed_fist': 'select',
            'peace_sign': 'next_page',
            'ok_sign': 'confirm',
            'swipe_left': 'previous',
            'swipe_right': 'next',
            'pinch': 'zoom_in',
            'spread': 'zoom_out',
            'wave': 'hello',
            'clap': 'attention'
        }
        
        # Colas para comunicación entre hilos
        self.voice_queue = queue.Queue()
        self.gesture_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Configuración de calibración
        self.calibration_data = {
            'voice_threshold': 0.7,
            'gesture_threshold': 0.8,
            'gesture_hold_time': 1.0,
            'voice_timeout': 5.0
        }
        
        print("🎤🤲 Sistema de Control por Voz y Gestos inicializado")
    
    def setup_voice_engine(self):
        """Configura el motor de síntesis de voz"""
        try:
            # Configurar voz
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Preferir voz femenina en español si está disponible
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Configurar velocidad y volumen
            self.tts_engine.setProperty('rate', 180)  # Palabras por minuto
            self.tts_engine.setProperty('volume', 0.8)  # Volumen (0.0 a 1.0)
            
            # Calibrar micrófono
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("🎤 Motor de voz configurado correctamente")
            
        except Exception as e:
            print(f"⚠️ Error configurando motor de voz: {e}")
    
    def start_control_system(self):
        """Inicia el sistema de control completo"""
        if self.is_active:
            return
        
        self.is_active = True
        
        # Iniciar hilos de control
        self.voice_thread = threading.Thread(target=self.voice_control_loop, daemon=True)
        self.gesture_thread = threading.Thread(target=self.gesture_control_loop, daemon=True)
        self.command_processor_thread = threading.Thread(target=self.command_processor_loop, daemon=True)
        
        self.voice_thread.start()
        self.gesture_thread.start()
        self.command_processor_thread.start()
        
        self.speak("Sistema de control por voz y gestos activado")
        print("🚀 Sistema de control iniciado")
    
    def stop_control_system(self):
        """Detiene el sistema de control"""
        self.is_active = False
        self.voice_active = False
        self.gesture_active = False
        
        self.speak("Sistema de control desactivado")
        print("🛑 Sistema de control detenido")
    
    def stop_voice_recognition(self):
        """Detiene solo el reconocimiento de voz"""
        self.voice_active = False
        print("🔇 Reconocimiento de voz detenido")
    
    def voice_control_loop(self):
        """Bucle principal de control por voz"""
        self.voice_active = True
        
        while self.is_active and self.voice_active:
            try:
                # Escuchar comando de voz
                command = self.listen_for_command()
                
                if command:
                    # Procesar comando
                    processed_command = self.process_voice_command(command)
                    
                    if processed_command:
                        # Añadir a cola de comandos
                        self.command_queue.put({
                            'type': 'voice',
                            'command': processed_command,
                            'raw_input': command,
                            'timestamp': datetime.now(),
                            'confidence': 0.8
                        })
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error en control de voz: {e}")
                time.sleep(1)
    
    def gesture_control_loop(self):
        """Bucle principal de control por gestos"""
        self.gesture_active = True
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️ No se pudo acceder a la cámara")
            self.gesture_active = False
            return
        
        while self.is_active and self.gesture_active:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Procesar frame para detección de gestos
                gesture_result = self.process_gesture_frame(frame)
                
                if gesture_result:
                    # Añadir a cola de comandos
                    self.command_queue.put({
                        'type': 'gesture',
                        'command': gesture_result['command'],
                        'gesture_type': gesture_result['gesture'],
                        'confidence': gesture_result['confidence'],
                        'timestamp': datetime.now(),
                        'coordinates': gesture_result.get('coordinates', {})
                    })
                
                # Mostrar frame con anotaciones (opcional)
                if hasattr(self, 'show_camera_feed') and self.show_camera_feed:
                    cv2.imshow('NeuroVision Gesture Control', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            except Exception as e:
                print(f"Error en control de gestos: {e}")
                time.sleep(0.1)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def command_processor_loop(self):
        """Procesa los comandos de la cola"""
        while self.is_active:
            try:
                if not self.command_queue.empty():
                    command_data = self.command_queue.get(timeout=1)
                    self.execute_command(command_data)
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error procesando comando: {e}")
    
    def listen_for_command(self) -> Optional[str]:
        """Escucha y reconoce comandos de voz"""
        try:
            with self.microphone as source:
                # Escuchar con timeout
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            # Reconocer voz
            try:
                # Intentar reconocimiento en español primero
                command = self.recognizer.recognize_google(audio, language='es-ES')
                return command.lower()
            except sr.UnknownValueError:
                # Intentar en inglés si falla español
                try:
                    command = self.recognizer.recognize_google(audio, language='en-US')
                    return command.lower()
                except sr.UnknownValueError:
                    return None
            
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"Error en reconocimiento de voz: {e}")
            return None
    
    def process_voice_command(self, command: str) -> Optional[str]:
        """Procesa y clasifica comandos de voz"""
        command = command.lower().strip()
        
        # Buscar coincidencias en comandos conocidos
        for action, phrases in self.voice_commands.items():
            for phrase in phrases:
                if phrase in command:
                    return action
        
        # Buscar comandos compuestos
        if 'mostrar' in command:
            if 'modelo' in command:
                return 'mostrar_modelo'
            elif 'datos' in command:
                return 'mostrar_datos'
            elif 'métricas' in command or 'metricas' in command:
                return 'mostrar_metricas'
        
        if 'crear' in command:
            if 'modelo' in command:
                return 'crear_modelo'
            elif 'gráfico' in command or 'grafico' in command:
                return 'crear_grafico'
        
        return None
    
    def process_gesture_frame(self, frame) -> Optional[Dict]:
        """Procesa un frame para detectar gestos"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Detectar manos
        hand_results = self.hands.process(rgb_frame)
        
        # Detectar pose
        pose_results = self.pose.process(rgb_frame)
        
        # Detectar rostro
        face_results = self.face_mesh.process(rgb_frame)
        
        # Analizar gestos de manos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = self.analyze_hand_gesture(hand_landmarks, w, h)
                if gesture:
                    return gesture
        
        # Analizar gestos de pose corporal
        if pose_results.pose_landmarks:
            gesture = self.analyze_pose_gesture(pose_results.pose_landmarks, w, h)
            if gesture:
                return gesture
        
        # Analizar gestos faciales
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                gesture = self.analyze_face_gesture(face_landmarks, w, h)
                if gesture:
                    return gesture
        
        return None
    
    def analyze_hand_gesture(self, landmarks, width: int, height: int) -> Optional[Dict]:
        """Analiza gestos de manos"""
        # Extraer coordenadas de puntos clave
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
        
        # Detectar gestos específicos
        gesture = self.detect_hand_gesture_pattern(points)
        
        if gesture:
            return {
                'gesture': gesture,
                'command': self.gesture_commands.get(gesture, 'unknown'),
                'confidence': 0.8,
                'coordinates': {
                    'hand_center': self.calculate_hand_center(points),
                    'fingers': self.get_finger_positions(points)
                }
            }
        
        return None
    
    def detect_hand_gesture_pattern(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """Detecta patrones específicos de gestos de mano"""
        if len(points) < 21:  # MediaPipe hand tiene 21 puntos
            return None
        
        # Índices de puntos clave de MediaPipe
        thumb_tip = points[4]
        thumb_ip = points[3]
        index_tip = points[8]
        index_pip = points[6]
        middle_tip = points[12]
        middle_pip = points[10]
        ring_tip = points[16]
        ring_pip = points[14]
        pinky_tip = points[20]
        pinky_pip = points[18]
        
        # Detectar dedos extendidos
        fingers_up = []
        
        # Pulgar
        if thumb_tip[0] > thumb_ip[0]:  # Pulgar derecho
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Otros dedos
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip[1] < pip[1]:  # Dedo extendido (y menor = más arriba)
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Clasificar gestos basado en dedos extendidos
        total_fingers = sum(fingers_up)
        
        if total_fingers == 0:
            return 'closed_fist'
        elif total_fingers == 5:
            return 'open_palm'
        elif fingers_up == [1, 0, 0, 0, 0]:
            return 'thumbs_up'
        elif fingers_up == [0, 1, 0, 0, 0]:
            return 'point_up'
        elif fingers_up == [0, 1, 1, 0, 0]:
            return 'peace_sign'
        elif fingers_up == [1, 1, 0, 0, 0]:
            return 'ok_sign'
        elif total_fingers == 1:
            return 'point_up'
        elif total_fingers == 2:
            return 'peace_sign'
        
        return None
    
    def analyze_pose_gesture(self, landmarks, width: int, height: int) -> Optional[Dict]:
        """Analiza gestos de pose corporal"""
        # Extraer puntos clave de pose
        pose_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            pose_points.append((x, y))
        
        # Detectar gestos de pose (brazos levantados, etc.)
        gesture = self.detect_pose_pattern(pose_points)
        
        if gesture:
            return {
                'gesture': gesture,
                'command': self.gesture_commands.get(gesture, 'unknown'),
                'confidence': 0.7,
                'coordinates': {
                    'pose_center': self.calculate_pose_center(pose_points)
                }
            }
        
        return None
    
    def detect_pose_pattern(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """Detecta patrones de pose corporal"""
        if len(points) < 33:  # MediaPipe pose tiene 33 puntos
            return None
        
        # Índices de puntos clave
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_elbow = points[13]
        right_elbow = points[14]
        left_wrist = points[15]
        right_wrist = points[16]
        
        # Detectar brazos levantados
        if (left_wrist[1] < left_shoulder[1] and 
            right_wrist[1] < right_shoulder[1]):
            return 'arms_up'
        
        # Detectar saludo con la mano
        if (left_wrist[1] < left_elbow[1] and 
            abs(left_wrist[0] - left_shoulder[0]) > 50):
            return 'wave'
        
        return None
    
    def analyze_face_gesture(self, landmarks, width: int, height: int) -> Optional[Dict]:
        """Analiza gestos faciales"""
        # Extraer puntos clave faciales
        face_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            face_points.append((x, y))
        
        # Detectar expresiones faciales básicas
        gesture = self.detect_face_expression(face_points)
        
        if gesture:
            return {
                'gesture': gesture,
                'command': self.gesture_commands.get(gesture, 'unknown'),
                'confidence': 0.6,
                'coordinates': {
                    'face_center': self.calculate_face_center(face_points)
                }
            }
        
        return None
    
    def detect_face_expression(self, points: List[Tuple[int, int]]) -> Optional[str]:
        """Detecta expresiones faciales básicas"""
        # Implementación básica - se puede expandir
        # Por ahora, detectar parpadeo como comando
        if len(points) > 100:
            # Análisis básico de ojos cerrados/abiertos
            # Esto es una simplificación
            return 'blink'
        
        return None
    
    def calculate_hand_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calcula el centro de la mano"""
        if not points:
            return (0, 0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return (center_x, center_y)
    
    def get_finger_positions(self, points: List[Tuple[int, int]]) -> Dict:
        """Obtiene posiciones de dedos"""
        if len(points) < 21:
            return {}
        
        return {
            'thumb': points[4],
            'index': points[8],
            'middle': points[12],
            'ring': points[16],
            'pinky': points[20]
        }
    
    def calculate_pose_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calcula el centro de la pose"""
        if len(points) < 2:
            return (0, 0)
        
        # Usar hombros como referencia
        left_shoulder = points[11] if len(points) > 11 else (0, 0)
        right_shoulder = points[12] if len(points) > 12 else (0, 0)
        
        center_x = (left_shoulder[0] + right_shoulder[0]) // 2
        center_y = (left_shoulder[1] + right_shoulder[1]) // 2
        
        return (center_x, center_y)
    
    def calculate_face_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calcula el centro de la cara"""
        if not points:
            return (0, 0)
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return (center_x, center_y)
    
    def execute_command(self, command_data: Dict):
        """Ejecuta un comando procesado"""
        try:
            command = command_data['command']
            command_type = command_data['type']
            
            print(f"🎯 Ejecutando comando {command_type}: {command}")
            
            # Confirmar comando por voz
            if command_type == 'voice':
                self.speak(f"Ejecutando {command}")
            
            # Ejecutar comando a través del callback handler
            if self.callback_handler:
                self.callback_handler(command_data)
            else:
                # Comandos básicos sin callback
                self.execute_basic_command(command, command_data)
            
        except Exception as e:
            print(f"Error ejecutando comando: {e}")
            self.speak("Error ejecutando comando")
    
    def execute_basic_command(self, command: str, command_data: Dict):
        """Ejecuta comandos básicos sin callback externo"""
        if command == 'ayuda':
            self.speak("Comandos disponibles: entrenar, mostrar, crear modelo, optimizar, evaluar")
        elif command == 'hello' or command == 'wave':
            self.speak("¡Hola! ¿En qué puedo ayudarte?")
        elif command == 'stop':
            self.speak("Comando de parada recibido")
        elif command == 'confirm':
            self.speak("Confirmado")
        else:
            self.speak(f"Comando {command} reconocido")
    
    def speak(self, text: str):
        """Síntesis de voz"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error en síntesis de voz: {e}")
    
    def set_callback_handler(self, handler: Callable):
        """Establece el manejador de callbacks para comandos"""
        self.callback_handler = handler
    
    def calibrate_voice(self):
        """Calibra el sistema de reconocimiento de voz"""
        self.speak("Iniciando calibración de voz. Di 'test' cuando estés listo.")
        
        for i in range(3):
            self.speak(f"Prueba {i + 1}. Di 'test'")
            command = self.listen_for_command()
            if command and 'test' in command:
                self.speak("Calibración exitosa")
                return True
        
        self.speak("Calibración fallida. Verifica tu micrófono.")
        return False
    
    def calibrate_gestures(self):
        """Calibra el sistema de reconocimiento de gestos"""
        self.speak("Iniciando calibración de gestos. Muestra tu mano abierta.")
        
        cap = cv2.VideoCapture(0)
        calibration_frames = 0
        successful_detections = 0
        
        while calibration_frames < 30:  # 30 frames de calibración
            ret, frame = cap.read()
            if not ret:
                continue
            
            gesture_result = self.process_gesture_frame(frame)
            if gesture_result and gesture_result['gesture'] == 'open_palm':
                successful_detections += 1
            
            calibration_frames += 1
            
            # Mostrar frame de calibración
            cv2.putText(frame, f"Calibracion: {calibration_frames}/30", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Calibracion de Gestos', frame)
            cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        success_rate = successful_detections / calibration_frames
        if success_rate > 0.7:
            self.speak("Calibración de gestos exitosa")
            return True
        else:
            self.speak("Calibración de gestos fallida. Verifica tu cámara.")
            return False
    
    def get_system_status(self) -> Dict:
        """Obtiene el estado del sistema de control"""
        return {
            'active': self.is_active,
            'voice_active': self.voice_active,
            'gesture_active': self.gesture_active,
            'voice_commands_available': len(self.voice_commands),
            'gesture_commands_available': len(self.gesture_commands),
            'last_gesture': self.gesture_state['last_gesture'],
            'gesture_confidence': self.gesture_state['gesture_confidence'],
            'calibration_data': self.calibration_data
        }

# Funciones de utilidad para integración

def create_voice_gesture_controller(callback_handler=None) -> VoiceGestureController:
    """Crea una instancia del controlador de voz y gestos"""
    return VoiceGestureController(callback_handler)

def start_voice_control(controller: VoiceGestureController):
    """Inicia solo el control por voz"""
    controller.voice_active = True
    controller.gesture_active = False
    controller.start_control_system()

def start_gesture_control(controller: VoiceGestureController):
    """Inicia solo el control por gestos"""
    controller.voice_active = False
    controller.gesture_active = True
    controller.start_control_system()

def start_full_control(controller: VoiceGestureController):
    """Inicia control completo por voz y gestos"""
    controller.voice_active = True
    controller.gesture_active = True
    controller.start_control_system()