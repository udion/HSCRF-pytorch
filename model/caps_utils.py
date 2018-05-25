import torch

def squash(s, dim=-1):
    '''
    "Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
    Args:
    s: 	Vector before activation
    dim:	Dimension along which to calculate the norm
    Returns:
    Squashed vector
    '''
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

def _caps_pri2sec_H(
        H,
        kernel_size,
        stride,
        padding,
        num_primary_caps,
    ):
    H1 = ((H - kernel_size + 2*padding)//stride) + 1
    sec_in = num_primary_caps*H1
    return sec_in

def _caps_pri2sec_HW(
        H,
        W,
        kernel_size,
        stride,
        padding,
        num_primary_caps,
    ):
    """
     TODO: handle the tuple versions of kernel_size, stride, padding
    """
    H1 = ((H - kernel_size + 2*padding)//stride) + 1
    W1 = ((W - kernel_size + 2*padding)//stride) + 1
    sec_in = num_primary_caps*H1*W1   
    return sec_in