def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def remove_pad_and_flat(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, K, L] or [B, K, L]
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 4:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 4:  # [B, C, K, L]
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 3:  # [B, K, L]
            results.append(input[:length].view(-1).cpu().numpy())
    return results
