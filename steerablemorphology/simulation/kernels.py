import torch
import torch.nn.functional as F

def generate_annular_kernel(radius: float, sigma: float, size: int = 31, device: str = "cpu") -> torch.Tensor:
    """
    Generates a 2D normalized annular (ring-shaped) Gaussian convolution kernel.
    
    Args:
        radius: Mean ring radius in grid units.
        sigma: Radial Gaussian standard deviation.
        size: Kernel matrix size (odd integer).
        device: Device to allocate tensor on.
        
    Returns:
        Normalized 2D torch.Tensor of shape (1, 1, size, size).
    """
    coords = torch.linspace(-(size // 2), size // 2, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    kernel = torch.exp(-0.5 * ((r - radius) / sigma) ** 2)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel.unsqueeze(0).unsqueeze(0)
