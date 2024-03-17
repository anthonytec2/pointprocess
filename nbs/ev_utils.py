import os
import pdb
import shutil
from functools import partial

import ffmpeg
import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import Video
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm


def load_dsec():
    f = h5py.File("../../data/events.h5")
    time_flow = np.loadtxt(
        "../../data/zurich_city_05_a_optical_flow_forward_timestamps.txt",
        skiprows=1,
        delimiter=",",
    )
    evs = f["events"]
    msidx = f["ms_to_idx"]
    t_offset = f["t_offset"]
    x = evs["x"]
    y = evs["y"]
    p = evs["p"]
    t = evs["t"]
    return x, y, p, t, t_offset


def none_safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if batch:
        from torch.utils.data.dataloader import default_collate

        return default_collate(batch)
    else:
        return {}


def init_weights(m):
    """Initialize weights according to the FlowNet2-pytorch from nvidia"""
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.0001, b=0.0001)
        nn.init.xavier_uniform_(m.weight, gain=0.001)

    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)


def num_trainable_parameters(module):
    trainable_parameters = filter(lambda p: p.requires_grad, module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])


def num_parameters(network):
    n_params = 0
    modules = list(network.modules())

    for mod in modules:
        parameters = mod.parameters()
        n_params += sum([np.prod(p.size()) for p in parameters])
    return n_params


def calc_floor_ceil_delta(x):
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]


def create_update(x, y, t, dt, p, vol_size, device="cpu"):
    assert (x >= 0).byte().all()
    assert (x < vol_size[2]).byte().all()
    assert (y >= 0).byte().all()
    assert (y < vol_size[1]).byte().all()
    assert (t >= 0).byte().all()
    # assert (t<vol_size[0] // 2).byte().all()

    if not (t < vol_size[0] // 2).byte().all():
        print(t[t >= vol_size[0] // 2])
        print(vol_size)
        raise AssertionError()

    vol_mul = torch.where(
        p < 0,
        torch.div(
            torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0],
            2,
            rounding_mode="floor",
        ),
        torch.zeros(p.shape, dtype=torch.long).to(device),
    )

    inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) + (vol_size[2]) * y + x

    vals = dt

    return inds, vals


def gen_discretized_event_volume(events, vol_size, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size).to(device)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] // 2 - 1) / (t_max - t_min + 1e-6))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    inds_fl, vals_fl = create_update(
        x, y, ts_fl[0], ts_fl[1], events[:, 3], vol_size, device=device
    )
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(
        x, y, ts_ce[0], ts_ce[1], events[:, 3], vol_size, device=device
    )

    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume


def create_update_xyt(x, y, t, dx, dy, dt, p, vol_size, device="cpu"):
    assert (x >= 0).byte().all()
    assert (x < vol_size[2]).byte().all()
    assert (y >= 0).byte().all()
    assert (y < vol_size[1]).byte().all()
    assert (t >= 0).byte().all()
    assert (t < vol_size[0]).byte().all()

    # vol_mul = torch.where(p < 0,
    #                      torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0] // 2,
    #                      torch.zeros(p.shape, dtype=torch.long).to(device))

    # only look at positive events
    vol_mul = 0
    inds = (vol_size[1] * vol_size[2]) * (t) + (vol_size[2]) * y + x

    vals = dx * dy * dt
    return inds, vals


def gen_discretized_event_volume_xyt(events, vol_size, weight=None, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size, device=device).half()

    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]

    # scale t
    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] - 1) / (t_max - t_min + 1e-7))

    # scale x and y
    x_scaled = x  # (x + 1e-8) * (vol_size[2] - 1)
    y_scaled = y  # (y + 1e-8) * (vol_size[1] - 1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    xs_fl, xs_ce = calc_floor_ceil_delta(x_scaled.squeeze())
    ys_fl, ys_ce = calc_floor_ceil_delta(y_scaled.squeeze())

    all_ts_options = [ts_fl, ts_ce]
    all_xs_options = [xs_fl, xs_ce]
    all_ys_options = [ys_fl, ys_ce]

    # interpolate in all three directions
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # choose a set of index and a set of interpolants
                x, y, t = (
                    all_xs_options[i][0],
                    all_ys_options[j][0],
                    all_ts_options[k][0],
                )
                dx, dy, dt = (
                    all_xs_options[i][1],
                    all_ys_options[j][1],
                    all_ts_options[k][1],
                )
                inds, vals = create_update_xyt(
                    x, y, t, dx, dy, dt, events[..., 3], vol_size, device=device
                )

                if weight is None:
                    volume.view(-1).put_(
                        inds.cuda(),
                        torch.tensor(events[:, 3]).cuda().half() * vals.cuda().half(),
                        accumulate=True,
                    )
                else:
                    volume.view(-1).put_(inds, vals * weight, accumulate=True)
    return volume


def create_update_evflow(x, y, t, dx, dy, dt, p, vol_size, device="cpu"):
    assert (x >= 0).byte().all()
    assert (x < vol_size[2]).byte().all()
    assert (y >= 0).byte().all()
    assert (y < vol_size[1]).byte().all()
    assert (t >= 0).byte().all()
    assert (t < vol_size[0]).byte().all()

    vol_mul = torch.where(
        (p < 0).to(device),
        torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0] // 2,
        torch.zeros(p.shape, dtype=torch.long).to(device),
    )

    # only look at positive events
    inds = (
        (vol_size[1] * vol_size[2]) * (t.to(device) + vol_mul)
        + (vol_size[2]) * y.to(device)
        + x.to(device)
    )

    vals = dx * dy * dt
    return inds, vals


def gen_discretized_evflow(events, vol_size, weight=None, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size).to(device)

    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]

    # scale t
    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] // 2 - 1) / (t_max - t_min + 1e-6))

    # scale x and y
    x_scaled = x  # (x + 1e-8) * (vol_size[2] - 1)
    y_scaled = y  # (y + 1e-8) * (vol_size[1] - 1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    xs_fl, xs_ce = calc_floor_ceil_delta(x_scaled.squeeze())
    ys_fl, ys_ce = calc_floor_ceil_delta(y_scaled.squeeze())

    all_ts_options = [ts_fl, ts_ce]
    all_xs_options = [xs_fl, xs_ce]
    all_ys_options = [ys_fl, ys_ce]

    # interpolate in all three directions
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # choose a set of index and a set of interpolants
                x, y, t = (
                    all_xs_options[i][0],
                    all_ys_options[j][0],
                    all_ts_options[k][0],
                )
                dx, dy, dt = (
                    all_xs_options[i][1],
                    all_ys_options[j][1],
                    all_ts_options[k][1],
                )

                inds, vals = create_update_evflow(
                    x, y, t, dx, dy, dt, events[..., 3], vol_size, device=device
                )
                if weight is None:
                    volume.view(-1).put_(
                        inds.to(device), vals.to(device), accumulate=True
                    )
                else:
                    volume.view(-1).put_(inds, vals * weight, accumulate=True)
    return volume


"""
 Network output is BxHxWxNx4, all between -1 and 1. Each 4-tuple is [x, y, t, p], 
 where [x, y] are relative to the center of the grid cell at that hxw pixel.
 This function scales this output to values in the range:
 [[0, volume_size[0]], [0, volume_size[1]], [0, volume_size[2]], [-1, 1]]
"""


def scale_events(events, volume_size, device="cuda"):
    # Compute the center of each grid cell.
    scale = volume_size[0] / events.shape[1]
    x_range = torch.arange(events.shape[2]).to(device) * scale + scale / 2
    y_range = torch.arange(events.shape[1]).to(device) * scale + scale / 2
    x_offset, y_offset = torch.meshgrid(x_range, y_range)

    t_scale = (volume_size[2] - 1) / 2.0
    # Offset the timestamps from [-1, 1] to [0, 2].
    t_offset = torch.ones(x_offset.shape).to(device) * t_scale
    p_offset = torch.zeros(x_offset.shape).to(device)
    offset = torch.stack(
        (x_offset.float(), y_offset.float(), t_offset, p_offset), dim=-1
    )
    offset = offset[None, ..., None, :]

    # Scale the [x, y] values to [-scale/2, scale/2] and
    # t to [-volume_size[2] / 2, volume_size[2] / 2].
    output_scale = (
        torch.tensor((scale / 2, scale / 2, t_scale, 1))
        .to(device)
        .reshape((1, 1, 1, 1, -1))
    )

    # Scale the network output
    events *= output_scale

    # Offset the network output
    events += offset

    events = torch.reshape(events, (events.shape[0], -1, 4))

    return events


def generate_random_samples(
    batch_size, radius, num_points, dim, inside=True, device="cpu"
):
    assert dim >= 1
    points = torch.empty(batch_size, num_points, dim).normal_()
    directions = points / points.norm(2, dim=2, keepdim=True)
    if inside:
        dist_center = torch.empty(batch_size, num_points, 1).uniform_() * radius
    else:
        dist_center = torch.ones(batch_size, num_points, 1)
    samples = directions * dist_center
    return samples.to(device)


def single_flow2rgb(flow_x, flow_y, hsv_buffer=None):
    if hsv_buffer is None:
        hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:, :, 1] = 1
    hsv_buffer[:, :, 0] = (np.arctan2(flow_y, flow_x) + np.pi) / (2.0 * np.pi)

    hsv_buffer[:, :, 2] = np.log1p(
        np.linalg.norm(np.stack((flow_x, flow_y), axis=0), axis=0)
    )
    flat = hsv_buffer[:, :, 2].reshape((-1))
    m = np.nanmax(flat[np.isfinite(flat)])
    if not np.isclose(m, 0.0):
        hsv_buffer[:, :, 2] /= m

    return colors.hsv_to_rgb(hsv_buffer)


def filter_ev_set(
    evs, x_b=(-1, -1), y_b=(-1, -1), t_b=(-1, -1), ms_idx=None, rescale=False
):
    """
    Function to filter event spatially
    evs(Nx4,(y,x,p,t): event set
    x_b(tuple 2 nums): (x_low, x_high) x_low<x<=x_high
    y_b(tuple 2 nums): (y_low, y_high) y_low<y<=y_high
    t_b(tuple 2 nums): (t_low, t_high) t_low<t<=t_high

    returns Kx4 Event Array, K<=N

    """

    assert evs.shape[1] == 4
    if not ms_idx is None and t_b:
        low = ms_idx[max(int(t_b[0] // 0.001 - 1), 0)]
        high = ms_idx[min(int(t_b[1] // 0.001 + 1), len(ms_idx) - 1)]

        if low == high:
            return np.zeros((2, 4))
        evs = evs[low:high]

    x_idx = (
        np.ones(evs.shape[0], dtype=bool)
        if x_b == (-1, -1)
        else (x_b[0] < evs[:, 1]) & (evs[:, 1] <= x_b[1])
    )
    y_idx = (
        np.ones(evs.shape[0], dtype=bool)
        if y_b == (-1, -1)
        else (y_b[0] < evs[:, 0]) & (evs[:, 0] <= y_b[1])
    )
    t_idx = (
        np.ones(evs.shape[0], dtype=bool)
        if t_b == (-1, -1)
        else (t_b[0] < evs[:, 3]) & (evs[:, 3] <= t_b[1])
    )

    # if x_b==(-1,-1) and t_b==(-1,-1):
    #     act_ind=x_idx&y_idx&t_idx
    # elif x_b==(-1,-1):
    #     act_ind=x_idx&y_idx
    # else:
    #     act_ind=t_idx
    ev_out = evs[x_idx & y_idx & t_idx]
    if rescale:
        ev_out[:, 0] -= max(y_b[0] + 1, 0)
        ev_out[:, 1] -= max(x_b[0] + 1, 0)
        ev_out[:, 3] -= max(t_b[0], 0)

    return ev_out


def save_fn(fun, cnt, val, out_folder, **kwargs):
    fun(val, **kwargs)
    plt.savefig(f"{out_folder}/{cnt:05}.jpg", dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close("all")


def plot(
    x, fun, out_folder="", delete=True, nb=True, quiet=True, out_name=None, fps=None, **kwargs
):
    """
    plot is a function that takes in a plotting function
    it will plot a for loop through each of the x values indicated
    you will pass all your keyword arguments through to this function
    after that we will plot in parallel all the values
    then save and create a movie from them
    Function needs arguments:
        cnt: index of what plot number
        val: value being looped over
        out_folder: where the data is output to
    TODO: Move saving logic into this function!
    """

    rnd_folder = int(np.random.random() * 100000)
    out_folder = out_folder if out_folder else f"imgs/{rnd_folder}"
    if os.path.exists(out_folder) and os.path.isdir(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    save_part = partial(save_fn, fun=fun)
    with Parallel(n_jobs=30, timeout=100, backend="multiprocessing") as parallel:
        r = parallel(
            delayed(save_part)(cnt=cnt, val=val, out_folder=out_folder, **kwargs)
            for cnt, val in enumerate(tqdm(x))
        )
    if not out_name:
        out_name = out_folder.split("/")[-1]
    if not os.path.exists("vids/") and not os.path.isdir("vids/"):
        os.makedirs("vids/")
    ffmpeg.input(f"{out_folder}/%05d.jpg").filter('pad', 'ceil(iw/2)*2', 'ceil(ih/2)*2').filter("fps", fps).output(
        f"vids/{out_name}.mp4",
        crf=23,
        y=None,
        vcodec="libx264",
        pix_fmt="yuv420p",
        movflags="faststart",
    ).run(quiet=True)
    

    
    print(f"Output to Folder: {out_name}")
    if delete:
        shutil.rmtree(out_folder)
    if nb:
        return Video(f"vids/{out_name}.mp4", embed=False, height=500)
    else:
        return


def ev_img(sub_evs, res=(480, 640)):
    img = np.zeros(res)
    img[sub_evs[:, 0].astype(np.uint16), sub_evs[:, 1].astype(np.uint16)] = sub_evs[
        :, 3
    ]
    return img


def single_flow2rgb2(flow_x, flow_y, hsv_buffer=None):
    if hsv_buffer is None:
        hsv_buffer = np.empty((flow_x.shape[0], flow_x.shape[1], 3))
    hsv_buffer[:, :, 1] = 1
    hsv_buffer[:, :, 0] = (np.arctan2(flow_y, flow_x) + np.pi) / (2.0 * np.pi)

    hsv_buffer[
        :, :, 2
    ] = 1  # np.log1p(np.linalg.norm(np.stack((flow_x, flow_y), axis=0), axis=0))
    flat = hsv_buffer[:, :, 2].reshape((-1))
    m = np.nanmax(flat[np.isfinite(flat)])
    if not np.isclose(m, 0.0):
        hsv_buffer[:, :, 2] /= m

    return colors.hsv_to_rgb(hsv_buffer)


def color_wheel():
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)

    xs, ys = np.meshgrid(x, y)

    rgb_flow = (single_flow2rgb(xs, ys) * 255).astype(np.uint8)
    return rgb_flow


if __name__ == "__main__":
    """
    events = torch.rand(3, 1000, 4).cuda()
    events[:, :, 3] = torch.rand(3, 1000).cuda() - 0.5
    vol_size = [3, 9, 100, 100]
    gen_batch_discretized_event_volume(events, vol_size)
    """
    events = torch.rand(1000, 4, requires_grad=True) * 100
    events[:, 2] = torch.arange(1000).float() / 1000
    events[:, 3] = torch.rand(1000) - 0.5
    events = torch.nn.parameter.Parameter(events)
    vol_size = [9, 100, 100]

    """
    import time
    t0 = time.time()
    print(time.time() - t0)
    mean = events.mean()
    mean.backward()
    print(events.grad)
    """
    optimizer = torch.optim.Adam([events], lr=0.1)
    for j in range(100):
        total_volume = []
        for i in range(8):
            event_volume = gen_discretized_event_volume(events, vol_size)
            total_volume.append(event_volume)
        total_volume = torch.stack(total_volume, axis=0)
        mean = events.mean()
        mean.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(mean.item())
