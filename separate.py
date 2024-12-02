import os
import shutil

# Directorio de origen de las carpetas y la carpeta 'seg' como destino
seg_dir = './image/seg'  # Carpeta de destino

# Asegúrate de que la carpeta 'seg' exista
if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)

# Bucle para recorrer las carpetas con el formato "UPENN-GBM-{num}_11"
for num in range(1, 630):  # De 00001 a 00611
    formatted_num = f"{num:05d}"  # Formatear a 5 dígitos
    folder_name = f"./image/UPENN-GBM-{formatted_num}_11"
    filename = f"UPENN-GBM-{formatted_num}_11_automated_approx_segm.nii.gz"
    
    source_path = os.path.join(folder_name, filename)
    destination_path = os.path.join(seg_dir, filename)
    
    # Verificar si el archivo existe en la carpeta de origen
    if os.path.exists(source_path):
        # Mover el archivo de vuelta a la carpeta 'seg'
        shutil.move(source_path, destination_path)
        print(f"Archivo {filename} movido de {folder_name} a {seg_dir}.")
    else:
        print(f"Archivo {filename} no encontrado en {folder_name}.")
