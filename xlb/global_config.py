class GlobalConfig:
    precision_policy = None
    velocity_set = None
    compute_backend = None


def init(velocity_set, compute_backend, precision_policy):
    GlobalConfig.velocity_set = velocity_set()
    GlobalConfig.compute_backend = compute_backend
    GlobalConfig.precision_policy = precision_policy


def current_backend():
    return GlobalConfig.compute_backend
