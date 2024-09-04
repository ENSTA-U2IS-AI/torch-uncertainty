def get_resnet_num_blocks(arch: int) -> list[int]:
    if arch == 18:
        num_blocks = [2, 2, 2, 2]
    elif arch == 20:
        num_blocks = [3, 3, 3]
    elif arch == 34 or arch == 50:
        num_blocks = [3, 4, 6, 3]
    elif arch == 44:
        num_blocks = [7, 7, 7]
    elif arch == 56:
        num_blocks = [9, 9, 9]
    elif arch == 101:
        num_blocks = [3, 4, 23, 3]
    elif arch == 110:
        num_blocks = [18, 18, 18]
    elif arch == 152:
        num_blocks = [3, 8, 36, 3]
    elif arch == 1202:
        num_blocks = [200, 200, 200]
    else:
        raise ValueError(f"Unknown ResNet architecture. Got {arch}.")
    return num_blocks
