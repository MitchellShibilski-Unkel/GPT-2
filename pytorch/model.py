import numpy as np
from torch import nn
from torch import Tensor
from torch import no_grad
from torch import range as trange
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
    
def split_states(x, n):
    # --- Reshape the last dimension of x into [n, x.shape[-1]/n] --- #
    *start, m = shapeList(x)
    return np.reshape(x, start + [n, m//n])

def merge_states(x):
    # --- Smash the last two dimensions of x into a single dimension --- #
    *start, a, b = shapeList(x)
    return np.reshape(x, start + [a * b])

def attention_mask(nd, ns, *, dtype):
    #* 1's in the lower triangle, counting from the lower right corner
    #! Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs
    
    i = trange(nd)[:, None]
    j = trange(ns)
    m = i >= j - ns + nd
    
    return Tensor(m).to(device).dtype(dtype)

def MLP(x, scope: int, n_state):
    for r in range(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2