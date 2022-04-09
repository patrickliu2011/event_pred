import torch

class Select(torch.nn.Module):
    """
    Selects channels by index along given dimension.
    
    Args:
        dim (int) - Dimension to select along
        indices (list of int) - Indices to keep
    """
    def __init__(self,
                 dim=-3,
                 indices=[0,1,2]):
        super(Select, self).__init__()
        self.dim = dim
        self.indices = indices
    
    def forward(self, x):
        x = x.transpose(0, self.dim)
        x = x[self.indices]
        x = x.transpose(0, self.dim)
        return x
   
class CustomScale(torch.nn.Module):
    """
    Applies affine transformation each value in a tensor, and applies a clamp.
    
    Args:
        scale (float): Value to multiply each tensor value by
        shift (float): Constant to add to scaled values
        clamp (2-tuple of float or None): Lower and upper bounds to clamp to,
            None for no clamping
    """
    def __init__(self,
                 scale=1/6400,
                 shift=0,
                 clamp=(0.0, 1.0)
                ):
        super(CustomScale, self).__init__()
        self.scale = scale
        self.shift = shift
        self.clamp = clamp
    
    def forward(self, x):
        return (x * self.scale + self.shift).clamp(self.clamp[0], self.clamp[1])