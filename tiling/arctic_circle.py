import enum
import itertools as it
import numpy as np
import evn
import tiling

# mode = 'torch'
mode = 'numpy'
# mode='cpp'

if mode == 'torch':
    import torch as th
elif mode == 'cpp':
    import arctic_circle_

class Cell(enum.IntEnum):
    U = 1
    D = 2
    L = 3
    R = 4
    EMPTY = 5
    ERROR = 6
    CORNER = 7

def main():
    arctic_circle_sanity_check()
    # show_arctic_circle_steps(4, seed=2)
    # show_arctic_circle_steps(4)
    # show_arctic_circle_pow2(8)
    make_arctic_circle_mov()

def make_arctic_circle_mov():
    # grids = [compute_arctic_circle_grids(100).grids[-1] for i in range(50)]
    grids = compute_arctic_circle_grids(1080).grids
    # grids = compute_arctic_circle_grids(540).grids
    print('make rgb', flush=True)
    imgs = [tiling.make_rgb_array(g, ARCTIC_CIRCLE_COLORS) for g in grids]
    imgs = [(img * 255).astype(np.uint8) for img in imgs]
    tiling.create_video_from_arrays(imgs, 'test.mp4')

def show_arctic_circle_steps(N, seed=-1):
    if seed > 0:
        np.random.seed(seed)
    info = compute_arctic_circle_grids(N, debug=True)
    for step in info.steps:
        add_UDLR_grig_to_plot(step)
    finish_UDLR_image_plot(info.grids[-1], save=False)

def show_arctic_circle_pow2(power2):
    N = 2**power2
    info = compute_arctic_circle_grids(N)
    for i in range(int(np.log2(N))):
        print(f'time iter {2**i:4} {info.times[2**i]*1000:7.3f}ms')
        add_UDLR_grig_to_plot(info.grids[2**i])
    finish_UDLR_image_plot(info.grids[-1])

def compute_arctic_circle_grids(N, debug=False):
    # np.random.seed(2)
    previous = fill_empty_rand(make_aztec_diamond(2))
    result = evn.Bunch(grids=[previous], times=[0], steps=[])
    for i in range(1, N):
        timer = evn.Chrono()
        pruned = remove_facing(previous)
        expanded = expand_grid(pruned)
        filled = fill_empty_rand(expanded)
        previous = filled
        result.grids.append(filled)
        result.times.append(timer.elapsed())
        if i%100 == 0: print(i, timer.elapsed()*1000, flush=True)
        if debug: result.steps.extend([pruned, expanded, filled])
    return result

def expand_grid(grid):
    expanded = make_aztec_diamond(len(grid) + 2)
    n = len(grid)
    expanded[expanded == Cell.EMPTY] = 0
    expanded[:n, 1:-1] += Cell.U * (grid == Cell.U)
    expanded[2:, 1:-1] += Cell.D * (grid == Cell.D)
    expanded[1:-1, :n] += Cell.L * (grid == Cell.L)
    expanded[1:-1, 2:] += Cell.R * (grid == Cell.R)
    expanded[expanded == 0] = Cell.EMPTY
    return expanded
    # for i, j in it.product(range(len(grid)), range(len(grid))):
    #     if grid[i, j] == U: expanded[i, j + 1:j + 3] = U
    #     if grid[i, j] == D: expanded[i + 2, j + 1:j + 3] = D
    #     if grid[i, j] == L: expanded[i + 1:i + 3, j] = L
    #     if grid[i, j] == R: expanded[i + 1:i + 3, j + 2] = R
    #     if grid[i, j] in (U, D): grid[i, j:j + 2] = EMPTY
    #     if grid[i, j] in (L, R): grid[i:i + 2, j] = EMPTY
    # return expanded

def remove_facing(grid):
    grid = tiling.copy_array(grid)
    okLR = (grid[:, 1:] != Cell.L) | (grid[:, :-1] != Cell.R)
    okUD = (grid[1:, :] != Cell.U) | (grid[:-1, :] != Cell.D)
    grid[:, 1:] *= okLR
    grid[:, :-1] *= okLR
    grid[1:, :] *= okUD
    grid[:-1, :] *= okUD
    grid[grid == 0] = Cell.EMPTY
    return grid
    # for i, j in it.product(range(len(grid) - 1), range(len(grid) - 1)):
    #     if grid[i, j] == D and grid[i + 1, j] == U:
    #         grid[i:i + 2, j:j + 2] = EMPTY
    #     if grid[i, j] == R and grid[i, j + 1] == L:
    #         grid[i:i + 2, j:j + 2] = EMPTY

def fill_empty_rand(grid):
    grid = tiling.copy_array(grid)
    emptysq = grid[:-1, :-1] == Cell.EMPTY
    emptysq &= grid[:-1, 1:] == Cell.EMPTY
    emptysq &= grid[1:, :-1] == Cell.EMPTY
    emptysq &= grid[:-1, :-1] == Cell.EMPTY
    for i in range(len(emptysq) - 1):
        emptysq[i + 1] = emptysq[i + 1] & ~emptysq[i]
        emptysq[:, i + 1] = emptysq[:, i + 1] & ~emptysq[:, i]
    emptysq = emptysq.to(int) if mode == 'torch' else emptysq.astype(int)
    if mode == 'torch':
        emptysq *= 1 + th.randint(2, size=emptysq.shape, device='cuda')
    else:
        emptysq *= 1 + np.random.randint(2, size=emptysq.shape)
    grid[grid == Cell.EMPTY] = 0
    UD = emptysq == 1
    LR = emptysq == 2
    Us, Ds = Cell.U * UD, Cell.D * UD
    Ls, Rs = Cell.L * LR, Cell.R * LR
    grid[:-1, :-1] += Us
    grid[:-1, 1:] += Us
    grid[1:, :-1] += Ds
    grid[1:, 1:] += Ds
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
    return grid

def make_aztec_diamond(n):
    assert n % 2 == 0
    if mode == 'torch':
        grid = Cell.EMPTY * th.ones((n, n), device='cuda', dtype=th.int16)
        J = th.arange(n)
        J = th.minimum(J, n - J - 1)
    else:
        grid = Cell.EMPTY * np.ones(n * n, dtype=np.int16).reshape(n, n)
        J = np.arange(n)
        J = np.minimum(J, n - J - 1)
    for i in range(n // 2):
        grid[i, i + J < n//2 - 1] = Cell.CORNER
        grid[n - i - 1, i + J < n//2 - 1] = Cell.CORNER
    return grid
    # for i in range(n // 2):
    #     for j in range(n // 2):
    #         if i + j < n//2 - 1:
    #             grid[i, j] = CORNER
    #             grid[n - i - 1, j] = CORNER
    #             grid[i, n - j - 1] = CORNER
    #             grid[n - i - 1, n - j - 1] = CORNER
    # return grid

ARCTIC_CIRCLE_COLORS = {
    Cell.U: (1, 0, 0),
    Cell.D: (1, 0.5, 0),
    Cell.L: (0, 1, 0),
    Cell.R: (0, 0, 1),
    Cell.EMPTY: (0.5, 0.5, 0.5),
    Cell.ERROR: (0, 0, 0),
    Cell.CORNER: (1, 1, 1),
}

def add_UDLR_grig_to_plot(grid):
    rgb = tiling.make_rgb_array(grid, ARCTIC_CIRCLE_COLORS)
    tiling.add_image_to_plot(rgb)

def finish_UDLR_image_plot(grid, save=True):
    rgb = tiling.make_rgb_array(grid, ARCTIC_CIRCLE_COLORS)
    if save: tiling.save_image_from_rgb_array(rgb, label='arctic_circle')
    tiling.add_image_to_plot(rgb)
    tiling.show_image_plot()

def arctic_circle_sanity_check():
    if mode == 'torch': return
    with evn.dev.temporary_random_seed(0):
        info = compute_arctic_circle_grids(6)
        golden = np.array([
            [7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 1, 1, 1, 1, 7, 7, 7, 7],
            [7, 7, 7, 1, 1, 1, 1, 1, 1, 7, 7, 7],
            [7, 7, 1, 1, 3, 4, 3, 2, 2, 4, 7, 7],
            [7, 3, 4, 3, 3, 4, 3, 1, 1, 4, 4, 7],
            [3, 3, 4, 3, 3, 2, 2, 2, 2, 4, 4, 4],
            [3, 3, 2, 2, 3, 1, 1, 3, 4, 4, 4, 4],
            [7, 3, 3, 2, 2, 4, 3, 3, 4, 4, 4, 7],
            [7, 7, 3, 3, 4, 4, 3, 3, 4, 4, 7, 7],
            [7, 7, 7, 3, 4, 2, 2, 3, 4, 7, 7, 7],
            [7, 7, 7, 7, 2, 2, 2, 2, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 2, 2, 7, 7, 7, 7, 7],
        ])
        assert np.all(info.grids[-1] == golden)

if __name__ == '__main__':
    main()
