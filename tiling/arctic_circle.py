import os
from time import perf_counter
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

matplotlib.use('TkAgg')
# USE_TORCH = True
USE_TORCH = False
if USE_TORCH:
    import torch as th

U = 2
D = -2
L = -1
R = 1
EMPTY = 3
ERROR = 4
CORNER = 5
_show_grids = []

def main():
    tstart = t = perf_counter()
    N = 512
    # np.random.seed(2)
    grid = make_aztec_diamond(2)
    fill_empty_rand(grid)
    for i in range(1, N):
        # if np.log2(i) % 1 == 0: show_grid(grid)
        remove_facing(grid)
        grid2 = expand_grid(grid)
        fill_empty_rand(grid2)
        grid = grid2
        if i % 100 == 0:
            print(f'{i:4} {perf_counter() - t:9.3f}', flush=True)
            t = perf_counter()
    print(f'generation time {perf_counter() - tstart:9.3f}')
    show_grid(grid)
    img = Image.fromarray((_show_grids[-1] * 255).astype(np.int8))
    img.convert("I;16").save(f'arctic_circle_{N}.png')
    # img.save(f'arctic_circle_{N}.png')
    finish_show()

def expand_grid(grid):
    n = len(grid)
    new = make_aztec_diamond(len(grid) + 2)
    # new = np.ones((len(grid)+2,len(grid)+2), dtype=np.int16) * CORNER
    new[new == EMPTY] = 0
    new[:n, 1:-1] += U * (grid == U)
    new[2:, 1:-1] += D * (grid == D)
    new[1:-1, :n] += L * (grid == L)
    new[1:-1, 2:] += R * (grid == R)
    new[new == 0] = EMPTY
    return new
    # for i, j in it.product(range(len(grid)), range(len(grid))):
    #     if grid[i, j] == U: new[i, j + 1:j + 3] = U
    #     if grid[i, j] == D: new[i + 2, j + 1:j + 3] = D
    #     if grid[i, j] == L: new[i + 1:i + 3, j] = L
    #     if grid[i, j] == R: new[i + 1:i + 3, j + 2] = R
    #     if grid[i, j] in (U, D): grid[i, j:j + 2] = EMPTY
    #     if grid[i, j] in (L, R): grid[i:i + 2, j] = EMPTY
    # return new

def remove_facing(grid):
    okLR = (grid[:, 1:] != L) | (grid[:, :-1] != R)
    grid[:, 1:] *= okLR
    grid[:, :-1] *= okLR
    okUD = (grid[1:, :] != U) | (grid[:-1, :] != D)
    # print(okUD)
    grid[1:, :] *= okUD
    grid[:-1, :] *= okUD
    grid[grid == 0] = EMPTY
    return
    # for i, j in it.product(range(len(grid) - 1), range(len(grid) - 1)):
    #     if grid[i, j] == D and grid[i + 1, j] == U:
    #         grid[i:i + 2, j:j + 2] = EMPTY
    #     if grid[i, j] == R and grid[i, j + 1] == L:
    #         grid[i:i + 2, j:j + 2] = EMPTY

def fill_empty_rand(grid):
    emptysq = grid[:-1, :-1] == EMPTY
    emptysq &= grid[:-1, 1:] == EMPTY
    emptysq &= grid[1:, :-1] == EMPTY
    emptysq &= grid[:-1, :-1] == EMPTY
    for i in range(len(emptysq) - 1):
        emptysq[i + 1] = emptysq[i + 1] & ~emptysq[i]
        emptysq[:, i + 1] = emptysq[:, i + 1] & ~emptysq[:, i]
    if USE_TORCH: emptysq = emptysq.to(int)
    else: emptysq = emptysq.astype(int)
    if USE_TORCH:
        emptysq *= 1 + th.randint(2, size=emptysq.shape, device='cuda')
    else:
        emptysq *= 1 + np.random.randint(2, size=emptysq.shape)
    grid[grid == EMPTY] = 0
    UD = emptysq == 1
    Us, Ds = U * UD, D * UD
    grid[:-1, :-1] += Us
    grid[:-1, 1:] += Us
    grid[1:, :-1] += Ds
    grid[1:, 1:] += Ds
    LR = emptysq == 2
    Ls, Rs = L * LR, R * LR
    grid[:-1, :-1] += Ls
    grid[:-1, 1:] += Rs
    grid[1:, :-1] += Ls
    grid[1:, 1:] += Rs
    # print(emptysq)
    # tmp = np.zeros_like(grid)
    # tmp[:-1,:-1] = emptysq
    # print(((grid==EMPTY)*2-tmp).astype(int))

    # for i, j in it.product(range(len(grid) - 1), range(len(grid) - 1)):
    #     if np.all(grid[i:i + 2, j:j + 2] == EMPTY):
    #         if np.random.rand() < 0.5:
    #             grid[i, j:j + 2] = U
    #             grid[i + 1, j:j + 2] = D
    #         else:
    #             grid[i:i + 2, j] = L
    #             grid[i:i + 2, j + 1] = R

def show_grid(grid):
    if USE_TORCH:
        _show_grids.append(grid.cpu().numpy())
    else:
        _show_grids.append(grid.copy())

def finish_show():
    global _show_grids
    _show_grids = _show_grids[-9:]
    n = int(np.ceil(np.sqrt(len(_show_grids))))
    fig, axes = plt.subplots(n, n)
    for i, grid in enumerate(_show_grids):
        ax = axes
        if n > 1: ax = axes[i // n, i % n]
        flat = grid.ravel()
        img = np.ones((flat.shape[0], 3))
        img[flat == U] = (1, 0, 0)
        img[flat == D] = (1, 0.5, 0)
        img[flat == L] = (0, 1, 0)
        img[flat == R] = (0, 0, 1)
        img[flat == EMPTY] = (0.5, 0.5, 0.5)
        img[flat == ERROR] = (0, 0, 0)
        img[flat == CORNER] = (1, 1, 1)
        ax.imshow(img.reshape(len(grid), len(grid), 3))
    for a in axes.ravel() if n > 1 else [axes]:
        a.axis('off')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    plt.show()

def make_aztec_diamond(n):
    assert n % 2 == 0
    if USE_TORCH:
        grid = EMPTY * th.ones((n, n), device='cuda', dtype=th.int16)
        J = th.arange(n)
        J = th.minimum(J, n - J - 1)
    else:
        grid = EMPTY * np.ones(n * n, dtype=np.int16).reshape(n, n)
        J = np.arange(n)
        J = np.minimum(J, n - J - 1)
    for i in range(n // 2):
        grid[i, i + J < n//2 - 1] = CORNER
        grid[n - i - 1, i + J < n//2 - 1] = CORNER
    return grid
    # for i in range(n // 2):
    #     for j in range(n // 2):
    #         if i + j < n//2 - 1:
    #             grid[i, j] = CORNER
    #             grid[n - i - 1, j] = CORNER
    #             grid[i, n - j - 1] = CORNER
    #             grid[n - i - 1, n - j - 1] = CORNER
    # return grid

if __name__ == '__main__':
    main()
