import sys
import os
import glob
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tiling

matplotlib.use('TkAgg')

def save_image_from_rgb_array(rgb, label='tiling', format='png'):
    assert len(rgb.shape) == 3 and rgb.shape[2] == 3
    rgb8 = (rgb * 255).astype(np.uint8)
    img = PIL.Image.fromarray(rgb8)
    fname_header = f'{tiling.projdir}/img/{label}/size{len(rgb)}/{label}_size{len(rgb)}'
    os.makedirs(os.path.dirname(fname_header), exist_ok=True)
    index = len(glob.glob(f'{fname_header}*'))
    img.save(f'{fname_header}_{index}.{format}')

_images_to_show = []

def make_rgb_array(grid, colors):
    flat = grid.ravel()
    if 'torch' in sys.modules:
        import torch
        img = torch.ones((flat.shape[0],3))
        for k, v in colors.items():
            img[flat == k] = torch.tensor(v, dtype=img.dtype)
    else:
        img = np.ones((flat.shape[0], 3))
        for k, v in colors.items():
            img[flat == k] = v
    img = img.reshape(len(grid), len(grid), 3)
    return img

def add_image_to_plot(rgb):
    USE_TORCH = False
    if 'torch' in sys.modules:
        import torch
        USE_TORCH = torch.is_tensor(rgb)
    if USE_TORCH:
        _images_to_show.append(rgb.cpu().numpy())
    else:
        _images_to_show.append(rgb.copy())

def show_image_plot():
    global _images_to_show
    _images_to_show = _images_to_show[-9:]
    n = int(np.ceil(np.sqrt(len(_images_to_show))))
    fig, axes = plt.subplots(n, n)
    for i, rgb in enumerate(_images_to_show):
        ax = axis if n == 1 else axes[i // n, i % n]
        ax.imshow(rgb)
    for a in axes.ravel() if n > 1 else [axes]:
        a.axis('off')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    plt.show()

def create_video_from_arrays(rgb_frames, fname, startscale=1.0):
    import skvideo.io, cv2
    n = rgb_frames[-1].shape[0]
    frames = []
    border_type = cv2.BORDER_CONSTANT
    border_color = [255, 255, 255]
    print('resize/pad frames', flush=True)
    startsize = int(startscale * n)
    for i, f in enumerate(rgb_frames):
        if len(f) < startsize:
            # f = cv2.resize(f, (n // 2, n // 2), interpolation=cv2.INTER_LINEAR)
            # f = cv2.resize(f, (startsize, startsize), interpolation=cv2.INTER_CUBIC)
            f = cv2.resize(f, (startsize, startsize), interpolation=cv2.INTER_NEAREST)
        udlr = [int(n - len(f)) // 2] * 4
        frames.append(cv2.copyMakeBorder(f, *udlr, border_type, value=border_color))
        # if i % 100 == 0: print(i, flush=True)
    for i in range(100):
        frames.append(frames[-1])
    writer = skvideo.io.FFmpegWriter(
        fname,
        outputdict={
            '-vcodec': 'libx264',  #use the h.264 codec
            # '-crf': '0',  #set the constant rate factor to 0, which is lossless
            # '-preset':'veryslow',  #the slower the better compression, in princple, try
            #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            # '-vf': f'scale={rgb_frames[0].shape[0]}:{rgb_frames[0].shape[1]}'
        })
    print('create video', flush=True)
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()
    print('    ...done')

