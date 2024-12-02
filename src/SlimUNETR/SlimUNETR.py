import torch
import nibabel as nib
import numpy as np

# Función para cargar imágenes NIfTI
def load_nifti_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

# Cargar las 4 imágenes
T1_path = 'D:/Luisa/luisa/Slim-UNETR-nuevo/Slim-UNETR-main/image/UPENN-GBM-00402_11/UPENN-GBM-00402_11_T1.nii.gz'
T1GD_path = 'D:/Luisa/luisa/Slim-UNETR-nuevo/Slim-UNETR-main/image/UPENN-GBM-00402_11/UPENN-GBM-00402_11_T1GD.nii.gz'
FLAIR_path = 'D:/Luisa/luisa/Slim-UNETR-nuevo/Slim-UNETR-main/image/UPENN-GBM-00402_11/UPENN-GBM-00402_11_FLAIR.nii.gz'
T2_path = 'D:/Luisa/luisa/Slim-UNETR-nuevo/Slim-UNETR-main/image/UPENN-GBM-00402_11/UPENN-GBM-00402_11_T2.nii.gz'

# Cargar las imágenes
image_T1 = load_nifti_image(T1_path)
image_T1GD = load_nifti_image(T1GD_path)
image_FLAIR = load_nifti_image(FLAIR_path)
image_T2 = load_nifti_image(T2_path)

# Verifica las dimensiones originales de las imágenes
print(f"Dimensiones originales de las imágenes (T1, T1GD, FLAIR, T2): {image_T1.shape}, {image_T1GD.shape}, {image_FLAIR.shape}, {image_T2.shape}")

# Apilar las imágenes en un tensor con 4 canales
stacked_images = torch.stack([torch.tensor(image_T1), torch.tensor(image_T1GD), torch.tensor(image_FLAIR), torch.tensor(image_T2)], dim=1)
print(f"Dimensiones después de apilar las imágenes: {stacked_images.shape}")

# Asegurarnos de que las dimensiones estén correctas (1, 4, W, H, D)
stacked_images = stacked_images.unsqueeze(0)  # Añadir batch size
print(f"Dimensiones después de añadir batch: {stacked_images.shape}")

# Redimensionar las imágenes (esto es para asegurar que el tamaño sea compatible con el modelo)
# El modelo espera un tensor con las dimensiones (batch_size, 4, W, H, D)
stacked_images = stacked_images.view(1, 4, 240, 240, 155)  # Las dimensiones deben ser 1, 4, 240, 240, 155
print(f"Dimensiones después de redimensionar: {stacked_images.shape}")

# Asegurarse de que la forma sea compatible con el modelo
# Ahora, stacked_images tiene la forma (1, 4, 240, 240, 155) que es lo que el modelo espera

# Cargar el modelo SlimUNETR
import torch.nn as nn
from src.SlimUNETR.Decoder import Decoder
from src.SlimUNETR.Encoder import Encoder

class SlimUNETR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=32,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        super(SlimUNETR, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

    def forward(self, x):
        embeding, hidden_states_out, (B, C, W, H, Z) = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out, (B, C, W, H, Z))
        return x

# Crear un tensor de entrada de prueba con las dimensiones esperadas
x = stacked_images  # Las imágenes preprocesadas con dimensiones correctas

# Inicializar el modelo
model = SlimUNETR(
    in_channels=4,
    out_channels=3,
    embed_dim=96,
    embedding_dim=32,
    channels=(24, 48, 60),
    blocks=(1, 2, 3, 2),
    heads=(1, 2, 4, 4),
    r=(4, 2, 2, 1),
    dropout=0.3,
)

# Ejecutar una pasada de prueba a través del modelo
output = model(x)
print(f"Salida del modelo: {output.shape}")
