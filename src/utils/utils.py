from mindspore.communication import get_rank


def is_main_process(n_gpu):
    if n_gpu > 1:
        return get_rank()
    else:
        return 0
