#!/usr/bin/env python3
"""
Lanzador de NeuroVision AI Interface
Script principal para iniciar la interfaz gráfica revolucionaria
"""

import sys
import os
import subprocess
import importlib.util

def check_dependencies():
    """Verifica las dependencias necesarias"""
    required_packages = [
        'tkinter',
        'customtkinter',
        'numpy',
        'matplotlib',
        'plotly',
        'tensorflow',
        'scikit-learn',
        'pandas',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                spec = importlib.util.find_spec(package)
                if spec is None:
                    missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Instala las dependencias faltantes"""
    print("🔧 Instalando dependencias...")
    
    # Instalar desde requirements
    requirements_path = os.path.join("interface", "requirements_interface.txt")
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_path
            ])
            print("✅ Dependencias instaladas correctamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
    else:
        print(f"❌ No se encontró {requirements_path}")
        return False

def setup_environment():
    """Configura el entorno"""
    # Añadir directorio de interfaz al path
    interface_dir = os.path.join(os.path.dirname(__file__), "interface")
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)
    
    # Configurar variables de entorno
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
    
    print("🔧 Entorno configurado")

def launch_interface():
    """Lanza la interfaz principal"""
    try:
        print("🚀 Iniciando NeuroVision AI Interface...")
        print("=" * 50)
        
        # Importar y ejecutar la interfaz
        from interface.main_interface import main
        main()
        
    except ImportError as e:
        print(f"❌ Error importando interfaz: {e}")
        print("Verifica que todos los archivos estén en su lugar")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando interfaz: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Función principal del lanzador"""
    print("🧠 NeuroVision AI Interface Launcher")
    print("=" * 40)
    
    # Verificar dependencias
    print("🔍 Verificando dependencias...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Dependencias faltantes: {', '.join(missing)}")
        print("🔄 Instalando dependencias automáticamente...")
        
        if not install_dependencies():
            print("❌ No se pudieron instalar las dependencias")
            return
    else:
        print("✅ Todas las dependencias están disponibles")
    
    # Configurar entorno
    setup_environment()
    
    # Lanzar interfaz
    if launch_interface():
        print("✅ Interfaz cerrada correctamente")
    else:
        print("❌ Error en la interfaz")

if __name__ == "__main__":
    main()