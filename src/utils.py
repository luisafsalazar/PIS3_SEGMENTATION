import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from accelerate import Accelerator
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from torch import nn
import datetime
import json
from sklearn.model_selection import train_test_split

# Función para determinar el dispositivo a usar (GPU si está disponible, sino CPU)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carga el state_dict de un modelo desde una URL o un archivo local y lo devuelve
def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    device = get_device()
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=device,
        )
    else:
        state_dict = torch.load(download_path, map_location=device)
    return state_dict

# Restaura el estado de entrenamiento desde el checkpoint más reciente, si existe
def resume_train_state(
    model,
    path: str,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    try:
        base_path = os.path.join(os.getcwd(), "model_store", path)
        dirs = [os.path.join(base_path, f.name) for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        accelerator.print(f"Intentando cargar {dirs[-1]} estado de entrenamiento")
        model = load_pretrain_model(os.path.join(dirs[-1], "pytorch_model.bin"), model, accelerator)
        starting_epoch = int(os.path.splitext(dirs[-1])[0].replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(f"Carga del estado de entrenamiento exitosa. Comenzando desde la época {starting_epoch}")
        return model, starting_epoch, step, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print("Fallo en la carga del estado de entrenamiento")
        return model, 0, 0, 0

# Carga un modelo preentrenado desde el path dado y lo asigna al modelo proporcionado
def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        model.load_state_dict(state_dict)
        accelerator.print("Modelo de entrenamiento cargado con éxito")
        return model.to(get_device())
    except Exception as e:
        accelerator.print(e)
        accelerator.print("Fallo en la carga del modelo de entrenamiento")
        return model

# Establece una semilla común para asegurar la reproducibilidad en CPU, GPU y numpy
def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

# Inicializa los pesos de las capas lineales y LayerNorm de un modelo
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

# Redirige la salida estándar y de error a la consola y a un archivo de registro si se especifica un directorio de logs
class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            self.log_file = open(os.path.join(logdir, 'log_file.txt'), 'w', encoding='utf-8')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        if self.console:
            sys.stdout = self.console  # Restaurar sys.stdout
            sys.stderr = sys.__stderr__  # Restaurar sys.stderr original
        if self.log_file is not None:
            self.log_file.close()

