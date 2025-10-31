import sys
import enum
import itertools
from multiprocessing import Pool
import numpy as np
import evn
import tiling

# MODE = 'loops'
# MODE = 'numpy'
# MODE = 'torch'
MODE = 'cpp'

def main():
    arctic_circle_sanity_check()
    # show_arctic_circle_steps(4, seed=1)
    #show_arctic_circle_steps(6)
    # with evn.Chrono():
        # show_arctic_circle_pow2(10, show=False)
    # make_arctic_circle_mov(540)
    make_arctic_circle_mov(1080)
    # evn.chronometer.report()

def make_aztec_diamond_jonah(n):
    grid = Cell.EMPTY * np.ones((n, n), dtype=np.uint8)
    n2 = n - 2
    n2 //= 2
    for row in range(n2):
        for col in range(n2):
            grid[row, col] = Cell.CORNER
            grid[n - row - 1, col] = Cell.CORNER
        for col in range(n - n2, n):
            grid[row, col] = Cell.CORNER
            grid[n - row - 1, col] = Cell.CORNER
        n2 -= 1
    return grid

def expand_grid_jonah(grid):
    n = len(grid)
    expanded = make_aztec_diamond_jonah(n + 2)
    for row in range(n):
        for col in range(n):
            if grid[row, col] == Cell.U:
                expanded[row, col+1] = Cell.U
            if grid[row, col] == Cell.D:
                expanded[row+2, col+1] = Cell.D
            if grid[row, col] == Cell.L:
                expanded[row+1, col] = Cell.L
            if grid[row, col] == Cell.R:
                expanded[row+1, col+2] = Cell.R

    return expanded

def remove_facing_jonah(grid):
    n = len(grid)
    for row in range(n-1):
        for col in range (n-1):
            if grid[row, col] == Cell.D and grid[row+1, col] == Cell.U:
                grid[row, col] = Cell.EMPTY
                grid[row+1, col] = Cell.EMPTY
            if grid[row, col] == Cell.R and grid[row, col+1] == Cell.L:
                grid[row, col] = Cell.EMPTY
                grid[row, col+1] = Cell.EMPTY

    return grid

def fill_empty_rand_jonah(grid):
    n = len(grid)
    for row in range(n-1):
        for col in range(n-1):
            if grid[row, col] == Cell.EMPTY:
                if np.random.randint(2):
                    grid[row, col] = Cell.U
                    grid[row, col+1] = Cell.U
                    grid[row+1, col] = Cell.D
                    grid[row+1, col+1] = Cell.D
                else:
                    grid[row, col] = Cell.L
                    grid[row+1, col] = Cell.L
                    grid[row, col+1] = Cell.R
                    grid[row+1, col+1] = Cell.R

    return grid

class Cell(enum.IntEnum):
    U = 1
    D = 2
    L = 3
    R = 4
    EMPTY = 5
    ERROR = 6
    CORNER = 7

if MODE == 'torch':
    import torch as th
elif MODE == 'cpp':
    import tiling.arctic_circle_compiled

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
        if i % 100 == 0: print(f'make grids {i:4} {timer.elapsed() * 1000:8.3f}ms', flush=True)
        if debug: result.steps.extend([pruned, expanded, filled])
    return result

@evn.chrono
def make_aztec_diamond(n):
    assert n % 2 == 0

    if MODE == 'cpp' and hasattr(tiling.arctic_circle_compiled, 'make_aztec_diamond'):
        grid = Cell.EMPTY * np.ones((n, n), dtype=np.uint8)
        return tiling.arctic_circle_compiled.make_aztec_diamond(grid)

    if MODE == 'loops':
        return make_aztec_diamond_jonah(n)
    elif MODE == 'torch':
        grid = Cell.EMPTY * th.ones((n, n), device='cuda', dtype=th.uint8)
        J = th.arange(n)
        J = th.minimum(J, n - J - 1)
    else:  # MODE == 'numpy':
        grid = Cell.EMPTY * np.ones((n, n), dtype=np.uint8)
        J = np.arange(n)
        J = np.minimum(J, n - J - 1)
    for i in range(n // 2):
        grid[i, i + J < n//2 - 1] = Cell.CORNER
        grid[n - i - 1, i + J < n//2 - 1] = Cell.CORNER
    return grid

@evn.chrono
def expand_grid(grid):
    n = len(grid)
    if MODE == 'cpp' and hasattr(tiling.arctic_circle_compiled, 'expand_grid'):
        expanded = make_aztec_diamond(n + 2)
        tiling.arctic_circle_compiled.expand_grid(grid, expanded)
        return expanded
    if MODE == 'loops':
        grid = grid.copy()
        return expand_grid_jonah(grid)
    else:
        expanded = make_aztec_diamond(n + 2)
        expanded[expanded == Cell.EMPTY] = 0
        expanded[:n, 1:-1] += tiling.cast_uint8(Cell.U * (grid == Cell.U))
        expanded[2:, 1:-1] += tiling.cast_uint8(Cell.D * (grid == Cell.D))
        expanded[1:-1, :n] += tiling.cast_uint8(Cell.L * (grid == Cell.L))
        expanded[1:-1, 2:] += tiling.cast_uint8(Cell.R * (grid == Cell.R))
        expanded[expanded == 0] = Cell.EMPTY
        return expanded

@evn.chrono
def remove_facing(grid):
    grid = tiling.copy_array(grid)

    if MODE == 'cpp' and hasattr(tiling.arctic_circle_compiled, 'remove_facing'):
        return tiling.arctic_circle_compiled.remove_facing(grid)
    if MODE == 'loops':
        return remove_facing_jonah(grid)
    else:
        okLR = (grid[:, 1:] != Cell.L) | (grid[:, :-1] != Cell.R)
        okUD = (grid[1:, :] != Cell.U) | (grid[:-1, :] != Cell.D)
        grid[:, 1:] *= okLR
        grid[:, :-1] *= okLR
        grid[1:, :] *= okUD
        grid[:-1, :] *= okUD
        grid[grid == 0] = Cell.EMPTY
        return grid

@evn.chrono
def fill_empty_rand(grid):
    grid = tiling.copy_array(grid)
    if MODE == 'cpp' and hasattr(tiling.arctic_circle_compiled, 'fill_empty_rand'):
        return tiling.arctic_circle_compiled.fill_empty_rand(grid)
    if MODE == 'loops':
        return fill_empty_rand_jonah(grid)
    else:
        emptysq = grid[:-1, :-1] == Cell.EMPTY
        emptysq &= grid[:-1, 1:] == Cell.EMPTY
        emptysq &= grid[1:, :-1] == Cell.EMPTY
        emptysq &= grid[:-1, :-1] == Cell.EMPTY
        for i in range(len(emptysq) - 1):
            emptysq[i + 1] = emptysq[i + 1] & ~emptysq[i]
            emptysq[:, i + 1] = emptysq[:, i + 1] & ~emptysq[:, i]
        emptysq = emptysq.to(int) if MODE == 'torch' else emptysq.astype(int)
        if MODE == 'torch':
            emptysq *= 1 + th.randint(2, size=emptysq.shape, device='cuda')
        else:
            emptysq *= 1 + np.random.randint(2, size=emptysq.shape)
        UD = emptysq == 1
        LR = emptysq == 2
        Us = Cell.U * UD
        Ds = Cell.D * UD
        Ls = Cell.L * LR
        Rs = Cell.R * LR
        if 'torch' not in sys.modules or not th.is_tensor(Us):
            Us, Ds, Ls, Rs = [_.astype(np.uint8) for _ in (Us, Ds, Ls, Rs)]
        grid[grid == Cell.EMPTY] = 0
        grid[:-1, :-1] += Us
        grid[:-1, 1:] += Us
        grid[1:, :-1] += Ds
        grid[1:, 1:] += Ds
        grid[:-1, :-1] += Ls
        grid[:-1, 1:] += Rs
        grid[1:, :-1] += Ls
        grid[1:, 1:] += Rs
        return grid

def _make_rgb(grid):
    img = tiling.make_rgb_array(grid, ARCTIC_CIRCLE_COLORS)
    return (img * 255).astype(np.uint8)

@evn.chrono
def make_arctic_circle_mov(n):
    # grids = [compute_arctic_circle_grids(100).grids[-1] for i in range(50)]
    grids = compute_arctic_circle_grids(n).grids
    # grids = compute_arctic_circle_grids(540).grids
    print('convert to rgb arrays', flush=True)
    # imgs = [_make_rgb(g) for g in grids]
    with Pool() as pool:
        imgs = pool.map(_make_rgb, grids)
    tiling.create_video_from_arrays(imgs, 'test.mp4')

def show_arctic_circle_steps(N, seed=-1):
    if seed >= 0: np.random.seed(seed)
    info = compute_arctic_circle_grids(N, debug=True)
    for step in info.steps:
        add_UDLR_grid_to_plot(step)
    finish_UDLR_image_plot(info.grids[-1], save=False)

def show_arctic_circle_pow2(power2, show=True):
    N = 2**power2
    info = compute_arctic_circle_grids(N)
    for i in range(int(np.log2(N))):
        print(f'time iter {2**i:4} {info.times[2**i]*1000:7.3f}ms')
        add_UDLR_grid_to_plot(info.grids[2**i])
    if show: finish_UDLR_image_plot(info.grids[-1])

ARCTIC_CIRCLE_COLORS = {
    Cell.U: (1, 0, 0),
    Cell.D: (1, 0.5, 0),
    Cell.L: (0, 1, 0),
    Cell.R: (0, 0, 1),
    Cell.EMPTY: (0.5, 0.5, 0.5),
    Cell.ERROR: (0, 0, 0),
    Cell.CORNER: (0.9, 0.9, 0.9),
}

def add_UDLR_grid_to_plot(grid):
    rgb = tiling.make_rgb_array(grid, ARCTIC_CIRCLE_COLORS)
    tiling.add_image_to_plot(rgb)

def finish_UDLR_image_plot(grid, save=True):
    rgb = tiling.make_rgb_array(grid, ARCTIC_CIRCLE_COLORS)
    if save: tiling.save_image_from_rgb_array(rgb, label='arctic_circle')
    tiling.add_image_to_plot(rgb)
    tiling.show_image_plot()

def arctic_circle_sanity_check():
    if MODE in 'torch loops cpp'.split(): return
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
