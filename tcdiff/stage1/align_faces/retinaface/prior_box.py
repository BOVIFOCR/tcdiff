from itertools import product
from math import ceil
from typing import List, Tuple

import torch


def priorbox(min_sizes: List[List[int]], steps: List[int], clip: bool, image_size: Tuple[int, int]) -> torch.Tensor:
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors: List[float] = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]


    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output
