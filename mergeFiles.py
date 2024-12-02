import os
import shutil

# Directorio de origen
source_dir = './image/seg'  # Carpeta de origen

# Bucle para recorrer los archivos con el formato "UPENN-GBM-{num}_11_automated_approx_segm.nii"
for num in range(1, 630):  # De 00001 a 00611
    formatted_num = f"{num:05d}"  # Formatear a 5 dígitos
    filename = f"UPENN-GBM-{formatted_num}_11_automated_approx_segm.nii.gz"
    
    # Carpeta de destino con el formato "UPENN-GBM-{num}_11"
    destination_dir = f"./image/UPENN-GBM-{formatted_num}_11"
    
    # Crear la carpeta de destino si no existe
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_dir, filename)
    
    # Verificar si el archivo existe en la carpeta de origen
    if os.path.exists(source_path):
        # Mover el archivo a la carpeta de destino
        shutil.move(source_path, destination_path)
        print(f"Archivo {filename} movido a {destination_dir}.")
    else:
        print(f"Archivo {filename} no encontrado en {source_dir}.")
