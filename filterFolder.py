import os
import shutil

# Directorio donde se encuentran las carpetas a evaluar
base_dir = './image'  # Reemplaza por la ruta de la carpeta donde están las subcarpetas

# Iterar sobre las carpetas dentro de base_dir
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Verificar si es una carpeta
    if os.path.isdir(folder_path):
        # Contar el número de archivos en la carpeta
        num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        # Si la carpeta tiene menos de 5 archivos, se elimina
        if num_files < 5:
            shutil.rmtree(folder_path)
            print(f"Carpeta {folder_name} eliminada. Tenía {num_files} archivos.")
        else:
            print(f"Carpeta {folder_name} no eliminada. Tenía {num_files} archivos.")
