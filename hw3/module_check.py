import torch
from torch import nn

import numpy as np
from utils import normal

FMT = {
    "OneLayerRNN": "{b}_fc.{a}",
    "MultiLayerRNN": "rnn.{c}.{b}_fc.{a}",
    "OneLayerLSTM": "{b}_fc.{a}",
    "MultiLayerLSTM": "lstm.{c}.{b}_fc.{a}"
}

class Info:
    def __init__(
            self,
            device,
            batch_size = 64,
            input_size = 128,
            hidden_size = 128,
            num_layers = 3,
            n_seq = 20,
            n_key = 10,
            embed_dim = 128,
            kdim = 64,
            vdim = 64,
            num_heads = 8
    ):
        self.device = device
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_seq = n_seq
        self.n_key = n_key
        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.num_heads = num_heads

def parameter_matching(model, target_model, model_type):
    fmt = FMT[model_type]

    param_dic = dict()
    for k, v in model.named_parameters():
        param_dic[k] = v

    for k, v in target_model.named_parameters():
        a, b, c = k.split("_")
        c = c[-1]
        param_dic[fmt.format(a = a, b = b, c = c)].data = v.data
    
def gradient_loss(model, target_model, model_type):
    fmt = FMT[model_type]

    grad_dic = dict()
    for k, v in model.named_parameters():
        grad_dic[k] = v.grad

    mse = nn.MSELoss()
    grad_loss = 0
    cnt = 0
    for k, v in target_model.named_parameters():
        cnt += 1
        a, b, c = k.split("_")
        c = c[-1]
        grad_loss += mse(v.grad, grad_dic[fmt.format(a = a, b = b, c = c)])

    print(f"Backward MSE\t\t\t: {grad_loss / cnt:.5e}")

def loss_rnn(out):
    return torch.norm(out[0]) + torch.norm(out[1])

def loss_lstm(out):
    x, (h, c) = out
    return torch.norm(x) + torch.norm(h) + torch.norm(c)

def check_one_layer_rnn(my_model, target_model, info: Info):

    model_type = "OneLayerRNN"
    batch_size = info.batch_size
    input_size = info.input_size
    hidden_size = info.hidden_size
    n_seq = info.n_seq
    device = info.device

    print(f"Model: {model_type}")
    print(f"batch_size: {batch_size}")
    print(f"input_size: {input_size}")
    print(f"hidden_size: {hidden_size}")
    print(f"n_seq: {n_seq}\n")

    parameter_matching(my_model, target_model, model_type)

    x = normal((batch_size, n_seq, input_size)).to(device)
    h = normal((batch_size, hidden_size)).to(device)

    out1, out2 = my_model(x, h), target_model(x, h.unsqueeze(0))

    mse = nn.MSELoss()

    print(f"Forward MSE for output layer\t: {mse(out1[0], out2[0]).item():.5e}")
    print(f"Forward MSE the last h\t\t: {mse(out1[1], out2[1].squeeze(0)).item():.5e}")
    

    loss1 = loss_rnn(out1)
    loss2 = loss_rnn(out2)

    loss1.backward()
    loss2.backward()

    gradient_loss(my_model, target_model, model_type)
    
def check_multi_layer_rnn(my_model, target_model, info: Info):

    model_type = "MultiLayerRNN"
    batch_size = info.batch_size
    input_size = info.input_size
    hidden_size = info.hidden_size
    num_layers = info.num_layers
    n_seq = info.n_seq
    device = info.device

    print(f"Model: {model_type}")
    print(f"batch_size: {batch_size}")
    print(f"input_size: {input_size}")
    print(f"hidden_size: {hidden_size}")
    print(f"num_layers: {num_layers}")
    print(f"n_seq: {n_seq}\n")

    parameter_matching(my_model, target_model, model_type)

    x = normal((batch_size, n_seq, input_size)).to(device)
    h = normal((num_layers, batch_size, hidden_size)).to(device)

    out1, out2 = my_model(x, h), target_model(x, h)

    mse = nn.MSELoss()

    print(f"Forward MSE for output layer\t: {mse(out1[0], out2[0]).item():.5e}")
    print(f"Forward MSE the last h\t\t: {mse(out1[1], out2[1]).item():.5e}")

    loss1 = loss_rnn(out1)
    loss2 = loss_rnn(out2)

    loss1.backward()
    loss2.backward()

    gradient_loss(my_model, target_model, model_type)

def check_one_layer_lstm(my_model, target_model, info: Info):

    model_type = "OneLayerLSTM"
    batch_size = info.batch_size
    input_size = info.input_size
    hidden_size = info.hidden_size
    n_seq = info.n_seq
    device = info.device

    print(f"Model: {model_type}")
    print(f"batch_size: {batch_size}")
    print(f"input_size: {input_size}")
    print(f"hidden_size: {hidden_size}")
    print(f"n_seq: {n_seq}\n")

    parameter_matching(my_model, target_model, model_type)

    x = normal((batch_size, n_seq, input_size)).to(device)
    h = normal((batch_size, hidden_size)).to(device)
    c = normal((batch_size, hidden_size)).to(device)

    out1, out2 = my_model(x, (h, c)), target_model(x, (h.unsqueeze(0), c.unsqueeze(0)))

    mse = nn.MSELoss()

    print(f"Forward MSE for output layer\t: {mse(out1[0], out2[0]).item():.5e}")
    print(f"Forward MSE the last h\t\t: {mse(out1[1][0], out2[1][0].squeeze(0)).item():.5e}")
    print(f"Forward MSE the last c\t\t: {mse(out1[1][1], out2[1][1].squeeze(0)).item():.5e}")
    

    loss1 = loss_lstm(out1)
    loss2 = loss_lstm(out2)

    loss1.backward()
    loss2.backward()

    gradient_loss(my_model, target_model, model_type)

def check_multi_layer_lstm(my_model, target_model, info: Info):

    model_type = "MultiLayerLSTM"
    batch_size = info.batch_size
    input_size = info.input_size
    hidden_size = info.hidden_size
    num_layers = info.num_layers
    n_seq = info.n_seq
    device = info.device

    print(f"Model: {model_type}")
    print(f"batch_size: {batch_size}")
    print(f"input_size: {input_size}")
    print(f"hidden_size: {hidden_size}")
    print(f"num_layers: {num_layers}")
    print(f"n_seq: {n_seq}\n")

    parameter_matching(my_model, target_model, model_type)

    x = normal((batch_size, n_seq, input_size)).to(device)
    h = normal((num_layers, batch_size, hidden_size)).to(device)
    c = normal((num_layers, batch_size, hidden_size)).to(device)
    init = (h, c)

    out1, out2 = my_model(x, init), target_model(x, init)

    mse = nn.MSELoss()

    print(f"Forward MSE for output layer\t: {mse(out1[0], out2[0]).item():.5e}")
    print(f"Forward MSE the last h\t\t: {mse(out1[1][0], out2[1][0]).item():.5e}")
    print(f"Forward MSE the last c\t\t: {mse(out1[1][1], out2[1][1]).item():.5e}")

    loss1, loss2 = loss_lstm(out1), loss_lstm(out2)

    loss1.backward()
    loss2.backward()

    gradient_loss(my_model, target_model, model_type)

def check_linear_projections(my_model, target_model, model_type):
    pass

def check_split_heads(my_model, target_model, model_type):
    pass

def check_scaled_dot_product_attention(my_model, target_model, model_type):
    pass

def check_multi_head_attention(my_model, target_model, info: Info):

    device = info.device
    batch_size = info.batch_size
    n_seq = info.n_seq
    n_key = info.n_key
    embed_dim = info.embed_dim
    kdim = info.kdim
    vdim = info.vdim
    num_heads = info.num_heads

    print("Model: MultiheadAttention")
    print(f"batch_size: {batch_size}")
    print(f"n_seq: {n_seq}")
    print(f"n_key: {n_key}")
    print(f"embed_dim: {embed_dim}")
    print(f"kdim: {kdim}")
    print(f"vdim: {vdim}")
    print(f"num_heads: {num_heads}\n")

    param_dic = dict()
    for name, param in my_model.named_parameters():
        param_dic[name] = param.data
    param_dic["in_proj_bias"] = torch.cat(
        (param_dic["q_proj.bias"],
        param_dic["k_proj.bias"],
        param_dic["v_proj.bias"])
    )

    for name, param in target_model.named_parameters():
        if "_weight" in name:
            manual_name = name.replace("_weight", ".weight")
        else:
            manual_name = name
        param.data = param_dic[manual_name]

    # Test input
    q = normal((batch_size, n_seq, embed_dim)).to(device)
    k = normal((batch_size, n_key, kdim)).to(device)
    v = normal((batch_size, n_key, vdim)).to(device)
    pad_mask = (torch.rand((batch_size, n_key)) > 0.5).to(device)

    # Forward check
    mse = nn.MSELoss()
    o, w = target_model(q, k, v, pad_mask)
    oo, ww = my_model(q, k, v, pad_mask)
    diff = mse(o, oo)
    print(f"Forward MSE for the output: {diff:.5e}")
    diff = mse(w, ww)
    print(f"Forward MSE for the attention weights: {diff:.5e}")

    # Backward check
    def loss_fn(out):
        return torch.norm(out[0]) + torch.norm(out[1])

    loss = loss_fn(target_model(q, k, v, pad_mask))
    manual_loss = loss_fn(my_model(q, k, v, pad_mask))

    loss.backward()
    manual_loss.backward()

    grad_dic = dict()
    for name, param in my_model.named_parameters():
        grad_dic[name] = param.grad
    grad_dic["in_proj_bias"] = torch.cat(
        (grad_dic["q_proj.bias"],
        grad_dic["k_proj.bias"],
        grad_dic["v_proj.bias"])
    )

    grad_loss = 0
    for name, param in target_model.named_parameters():
        if "_weight" in name:
            manual_name = name.replace("_weight", ".weight")
        else:
            manual_name = name
        grad_loss += mse(param.grad, grad_dic[manual_name])

    print(f"Backward MSE: {grad_loss:.5e}")







