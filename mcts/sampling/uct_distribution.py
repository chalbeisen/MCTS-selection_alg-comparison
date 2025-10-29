import torch


def uct_distribution(puct_values: torch.Tensor, inf_value_cap_coeff: float = 10.0) -> torch.Tensor:
    """
    UCT distribution. UCT values can be any real number (e.g. -inf).
    """
    # treat infinities
    #sorted_puct = puct_values.sort().values
    infinite_puct = puct_values.abs() == float("inf")
    # normalize only if there are values that are not inf
    if len(puct_values[infinite_puct]) != len(puct_values):
        largest_finite_puct = puct_values[~infinite_puct].abs().max()
        if largest_finite_puct > 0: 
            capped_puct = puct_values.clone() / largest_finite_puct
        else:
            capped_puct = puct_values.clone()
        capped_puct[infinite_puct] = inf_value_cap_coeff * torch.sign(puct_values[infinite_puct])
    else:
        capped_puct = inf_value_cap_coeff * torch.sign(puct_values)

    # softmax
    puct_probabilities = torch.softmax(capped_puct, dim=0)

    return puct_probabilities


if __name__ == "__main__":
    from cap_distribution import cap_distribution

    uct_values = torch.tensor([-float("inf"), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")])
    dist = uct_distribution(uct_values, 2)
    print(dist)
    res = cap_distribution(dist, 0.5)
    print(res)
