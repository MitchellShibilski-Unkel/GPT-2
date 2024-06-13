import numpy as np
from torch import nn
from torch import Tensor
from torch import no_grad
from torch import zeros, ones, sqrt


GPU: bool = False
CPU: bool = True
device: str = 'cuda' if GPU else 'cpu'

def default_params():
    n_vocab, n_ctx, n_embd, n_head, n_layer = 0, 1024, 768, 12, 12
    return n_vocab, n_ctx, n_embd, n_head, n_layer

def sendToGPU():
    GPU, CPU = True, False
    
def sendToCPU():
    GPU, CPU = False, True

def shapeList(x):
    shape = np.array([x]).reshape(-1, 1).shape
    tensor = Tensor(shape).to(device)
    return tensor

def softmax(x, axis=-1):
    prediction = nn.Softmax(axis).to(device)
    return prediction(x)

def gelu(x):
    return nn.GELU().to(device)(x)

def conv1d(x, firstLayerNum: int = 16, secondLayerNum: int = 8, size: int = 1):
    return nn.Conv1d(firstLayerNum, secondLayerNum, size).to(device)(x)

def norm(x, *, axis=-1, epsilon=1e-5):
    # --- Normalize to mean = 0, std = 1, then do a diagonal affine transform --- #
    with no_grad():
        n_state = x.shape[-1]
        g = nn.Parameter(ones(n_state))
        b = nn.Parameter(zeros(n_state))
        u = x.mean(axis=axis, keepdim=True)
        s = ((x - u) ** 2).mean(axis=axis, keepdim=True)
        x = (x - u) / sqrt(s + epsilon) * g + b
        return x
