import torch
from torch.autograd import Variable
from caps_utils import _caps_pri2sec_H
from caps import Caps1d_primary, Caps1d_secondary

w_s_len = 10
batch_size = 4

p_in_ch = 1
n_p_caps = 4
p_caps_dim = 16
H = 16

P1 = Caps1d_primary(p_in_ch, n_p_caps, p_caps_dim)
sec_in = _caps_pri2sec_H(H, 3, 1, 1, 4)

n_sec_caps=2
sec_caps_dim=16

S1 = Caps1d_secondary(sec_in, p_caps_dim, n_sec_caps, sec_caps_dim)


x = Variable(torch.rand(w_s_len*batch_size, 1, 16))

x1 = P1(x)
x1 = S1(x1)

print(x.size())
