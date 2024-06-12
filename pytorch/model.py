import numpy as np
from torch import nn
from torch import Tensor


GPU: bool = False
CPU: bool = True
device: str = 'cuda' if GPU else 'cpu'

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