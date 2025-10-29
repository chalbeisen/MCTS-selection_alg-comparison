import torch


def sample_from_discrete_distribution(weights: torch.Tensor, num_samples: int = 1, eps: float = 1e-6) -> int:
    """
    Sample from a discrete distribution given probability weights.

    Args:
        weights (torch.Tensor): A 1D tensor containing probability weights that sum to 1

    Returns:
        int: The sampled index from the distribution

    Example:
        >>> weights = torch.tensor([0.2, 0.3, 0.5])
        >>> sampled_idx = sample_from_discrete_distribution(weights)
    """
    # Ensure weights sum to 1 by normalizing
    all_weights = torch.sum(weights)
    if not torch.isclose(all_weights, torch.tensor(1.0), atol=eps):
        normalized_weights = weights / all_weights

    # Sample from the distribution
    sampled_idx = torch.multinomial(normalized_weights, num_samples=num_samples).item()

    return sampled_idx
