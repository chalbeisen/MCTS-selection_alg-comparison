import torch

def uct_distribution(uct_values: torch.Tensor, inf_value_cap_coeff: float = 10.0) -> torch.Tensor:
    """
    UCT distribution. UCT values can be any real number (e.g. -inf).
    Args:
        uct_values (torch.Tensor): UCT values.
        inf_value_cap_coeff (float): Replacement value infinite values.
    """
    # treat infinities
    inf_mask = torch.isinf(uct_values)

    if inf_mask.all():
        capped = inf_value_cap_coeff * torch.sign(uct_values)
        return torch.softmax(capped, dim=0).float()

    finite_vals = uct_values[~inf_mask]

    max_abs = finite_vals.abs().max()

    capped = uct_values.clone()

    if max_abs > 0:
        capped[~inf_mask] /= max_abs

    capped[inf_mask] = inf_value_cap_coeff * torch.sign(uct_values[inf_mask])

    return torch.softmax(capped, dim=0).float()


if __name__ == "__main__":
    from cap_distribution import cap_distribution

    uct_values = torch.tensor([-float("inf"), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")])
    dist = uct_distribution(uct_values, 2)
    print(dist)
    res = cap_distribution(dist, 0.5)
    print(res)
