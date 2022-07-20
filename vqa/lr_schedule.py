"""Learning rate utilities."""

def linear_warmup(warmup_steps, current_step):
    return min([1.0, float(current_step)/float(warmup_steps)])

def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5

def rsqrt_hidden(hidden_size):
    return float(hidden_size) ** -0.5

def create_dynamic_lr(schedule, training_steps, learning_rate, warmup_steps,
                      start_decay_step=0, min_lr=0.):
    """
    Generate dynamic learning rate.
    """
    if start_decay_step < warmup_steps:
        start_decay_step = warmup_steps
    lr = []
    for current_step in range(1, training_steps+1):
        cur_lr = 1.0
        for name in schedule.split("*"):
            if name == "constant":
                cur_lr *= float(learning_rate)
            elif name == "rsqrt_hidden":
                cur_lr *= rsqrt_hidden(hidden_size)
            elif name == "linear_warmup":
                cur_lr *= linear_warmup(warmup_steps, current_step)
            elif name == "rsqrt_decay":
                cur_lr *= rsqrt_decay(warmup_steps, current_step-start_decay_step+warmup_steps)
            else:
                raise ValueError("unknown learning rate schedule")
        if warmup_steps < current_step < start_decay_step:
            cur_lr = lr[-1]
        if current_step > warmup_steps:
            cur_lr = max([cur_lr, min_lr])
        lr.append(cur_lr)
    return lr