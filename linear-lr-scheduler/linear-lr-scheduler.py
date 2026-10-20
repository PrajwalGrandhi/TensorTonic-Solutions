def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if step < warmup_steps:
        return step * initial_lr / warmup_steps
    elif warmup_steps <= step <= total_steps:
        if total_steps - warmup_steps > 0:
            return final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps)
        else:
            return final_lr
    else:
        return final_lr