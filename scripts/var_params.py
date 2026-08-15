import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import distributed as dist

import matplotlib.pyplot as plt
import os
import random
import yaml
import time
def plot_comparison_grid(
    p_gt: np.ndarray,
    u_gt: np.ndarray,
    v_gt: np.ndarray,
    p_pred: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    save_dir: str,
    offset: int,
    fname_suffix: str = "",
):

    # --- Prepare figure ---
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), constrained_layout=True)

    # Top row: Ground truth
    im1 = axes[0, 0].imshow(p_gt, origin="lower", cmap="viridis")
    axes[0, 0].set_title(r"$p_{\mathrm{True}}$", fontsize=12)
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(u_gt, origin="lower", cmap="viridis")
    axes[0, 1].set_title(r"$u_{\mathrm{True}}$", fontsize=12)
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[0, 2].imshow(v_gt, origin="lower", cmap="viridis")
    axes[0, 1].set_title(r"$v_{\mathrm{True}}$", fontsize=12)
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom row: Predicted
    im4= axes[1, 0].imshow(p_pred, origin="lower", cmap="viridis")
    axes[1, 0].set_title(r"$p_{\mathrm{Predicted}}$", fontsize=12)
    fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(u_pred, origin="lower", cmap="viridis")
    axes[1, 1].set_title(r"$u_{\mathrm{Predicted}}$", fontsize=12)
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(v_pred, origin="lower", cmap="viridis")
    axes[1, 2].set_title(r"$v_{\mathrm{Predicted}}$", fontsize=12)
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # --- Format and save ---
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plot_path = os.path.join(save_dir, "plot")
    os.makedirs(plot_path, exist_ok=True)

    filename = f"sample_{offset}{fname_suffix}.png"
    fig.savefig(os.path.join(plot_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_and_plot_metrics(
    save_dir: str,
    offset: int,
    relative_error_p: float,
    relative_error_u: float,
    relative_error_v: float,
    TypeProblem: str,
):
    """
    Save current metrics (offset, relative errors) to metrics.npy
    and generate a color-coded plot of relative errors for u and v.

    Args:
        save_dir (str): Directory where metrics.npy and plots will be saved.
        offset (int): Data sample offset.
        relative_error_u (float): Relative L2 error for u.
        relative_error_v (float): Relative L2 error for v.
        TypeProblem (str): 'forward' or 'inverse'.
    """

    # --- Save or append metrics file ---
    output_path = os.path.join(save_dir, "metrics.npy")

    # Reset file on first offset
    if offset == 0 and os.path.exists(output_path):
        os.remove(output_path)

    # Load existing or start new
    if os.path.exists(output_path):
        metrics_values = np.load(output_path, allow_pickle=True).tolist()
    else:
        metrics_values = []

    entry = {
        "offset": int(offset),
        "relative_error_p": float(relative_error_p),
        "relative_error_u": float(relative_error_u),
        "relative_error_v": float(relative_error_v),
    }
    metrics_values.append(entry)

    np.save(output_path, np.array(metrics_values, dtype=object), allow_pickle=True)

    # --- Prepare arrays for plotting ---
    metrics = np.load(output_path, allow_pickle=True).tolist()
    offsets = np.array([e["offset"] for e in metrics])
    errors_p = np.array([e["relative_error_p"] for e in metrics])
    errors_u = np.array([e["relative_error_u"] for e in metrics])
    errors_v = np.array([e["relative_error_v"] for e in metrics])

    # Sort by offset for clean plots
    order = np.argsort(offsets)
    offsets, errors_u, errors_v = offsets[order], errors_u[order], errors_v[order]

    # --- Plot configuration ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    titles = ["Relative Error (p)", "Relative Error (u)", "Relative Error (v)"]
    series = [errors_p, errors_u, errors_v]

    if TypeProblem == "forward":
        styles = [("tab:red", "dotted"), ("tab:red", "dotted"),("tab:blue", "solid")]   # v highlighted
    elif TypeProblem == "inverseu":
        styles = [("tab:red", "dotted"), ("tab:blue", "solid"),("tab:red", "dotted")]   # u highlighted
    elif TypeProblem == "inversep":
        styles = [("tab:blue", "solid"), ("tab:red", "dotted"),("tab:red", "dotted")]   # p highlighted
    else:
        raise ValueError(f"Unknown TypeProblem: {TypeProblem}")

    # --- Plotting ---
    for ax, vals, title, (color, linestyle) in zip(axes, series, titles, styles):
        mean_val, std_val = np.mean(vals), np.std(vals)
        ax.plot(
            offsets,
            vals,
            marker="o",
            linestyle=linestyle,
            linewidth=1.8,
            color=color,
            label=title,
        )
        ax.set_xlabel("Offset")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"{title}\nMean={mean_val:.4f}, Std={std_val:.4f}")
        ax.grid(True, alpha=0.4)
        ax.legend()

    # --- Save and close ---
    plot_path = os.path.join(save_dir, "metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalize_field(x, stats):
    return ((x -stats["b"]) / stats["a"]).to(torch.float64)

def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    rng = np.random.default_rng(seed=seed)
    indices =rng.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


    
def var_params(config, offset, typePDE, Nobs, TypeProblem, seed, device_index):
    

    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    save_dir = config['data']['save_dir']

    device = f'cuda:{device_index}'
    save_dir = os.path.join(save_dir, 'Nobs_' + str(Nobs))
    save_dir = os.path.join(save_dir,  typePDE)
    save_dir = os.path.join(save_dir,  TypeProblem)

    os.makedirs(save_dir, exist_ok=True) 
    # load the data

    if typePDE == 'diffusion':
        npz_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if not f.endswith("adve_diff.npz") and f.endswith("diff.npz") ]
    elif typePDE == 'advection':
        npz_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith("adve.npz") ]
    elif typePDE == 'advection_diffusion':
        npz_files = [os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith("adve_diff.npz") ]
    else:
        raise ValueError(f'Type of PDE {typePDE} not found')


    datapoint = npz_files[offset]
    fname = datapoint

    with open(fname, "rb") as f:
        data = np.load(f)
        label = int(data["label"])
        init_value = data["init_value"].reshape(64, 64)
        last_value = data["last_value"].reshape(64, 64)

        param_key = {0: "diffusivity", 1: "velocity_x", 2: "diffusivity"}.get(label)
        if param_key is None:
            raise ValueError(f"Invalid label {label}. Must be 0, 1, or 2.")

        param_value = data[param_key]

    param_field = param_value * np.ones_like(init_value)
    image = np.stack((param_field, init_value, last_value), axis=-1)
    image = image.astype(np.float64)


    # Convert stacked image channels into torch tensors
    p_GT = torch.tensor(image[:, :, 0], dtype=torch.float64, device=device)
    u_GT = torch.tensor(image[:, :, 1], dtype=torch.float64, device=device)
    v_GT = torch.tensor(image[:, :, 2], dtype=torch.float64, device=device)


    batch_size = config['generate']['batch_size']

    # set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("*"*170)
    
    # load the network    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)

    # Load stats
    with open("Stats.yaml", "r") as f:
        stats_cfg = yaml.safe_load(f)

    p_stats = stats_cfg["var_params"]["p"][typePDE]
    u_stats = stats_cfg["var_params"]["u"]
    v_stats = stats_cfg["var_params"]["v"]

    # print the information
    print('='*170)
    print(f"[INFO] PDE = {typePDE:<12}  Problem = {TypeProblem:<8}  Offset = {offset:<3}"
    f"Label = {label:<2}  Nobs = {Nobs:<5}  Seed = {seed:<4}  "
    f"Device = {device}")
    print('='*170)

    try:
        zeta_cfg = config["guidance_map"][TypeProblem][typePDE]

        zeta_obs_p = zeta_cfg.get("zeta_obs_p")
        zeta_obs_u = zeta_cfg.get("zeta_obs_u")
        zeta_obs_v = zeta_cfg.get("zeta_obs_v")

        print(f"{TypeProblem.capitalize()} problem -> zeta_p = {zeta_obs_p}, zeta_u = {zeta_obs_u}, zeta_v = {zeta_obs_v}")

        print("=" * 170)

    except KeyError as e:
        raise ValueError(
            f"Invalid combination in guidance_map: (TypeProblem={TypeProblem}, PDE={typePDE})"
        ) from e
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')

    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

    class_labels = torch.eye(3, device=device)[label]

    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    known_index_u = random_index(Nobs, 64, seed=offset+1, device=device)
    known_index_v = random_index(Nobs, 64, seed=offset+2, device=device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sampling_start = time.perf_counter()


    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Scale the data back
        p_N = x_N[:,0,:,:].unsqueeze(0)
        u_N = x_N[:,1,:,:].unsqueeze(0)
        v_N = x_N[:,2,:,:].unsqueeze(0)

        p_N = normalize_field(p_N, p_stats)
        u_N = normalize_field(u_N, u_stats)
        v_N = normalize_field(v_N, v_stats)
       
        observation_loss_p = (p_N - p_GT).squeeze()
        observation_loss_u = (u_N - u_GT).squeeze()*known_index_u
        observation_loss_v = (v_N - v_GT).squeeze()*known_index_v

        L_obs_p = torch.norm(observation_loss_p, 2)/float(64*64)
        L_obs_u = torch.norm(observation_loss_u, 2)/float(Nobs)
        L_obs_v = torch.norm(observation_loss_v, 2)/float(Nobs)


        grad_x_cur_obs_p = torch.autograd.grad(outputs=L_obs_p, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_v = torch.autograd.grad(outputs=L_obs_v, inputs=x_cur, retain_graph=True)[0]

        x_next = x_next  -zeta_obs_u * grad_x_cur_obs_u -zeta_obs_v * grad_x_cur_obs_v -zeta_obs_p * grad_x_cur_obs_p
    ############################ Save the data ############################
    # CUDA operations are asynchronous, so this synchronization is essential.
    torch.cuda.synchronize(device)

    sampling_time = time.perf_counter() - sampling_start

    peak_memory_allocated_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )

    peak_memory_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / 1024**3
    )

    print("=" * 170)
    print(f"Posterior sampling time: {sampling_time:.2f} s")
    print(
        f"Peak allocated GPU memory: "
        f"{peak_memory_allocated_gb:.3f} GB"
    )
    print(
        f"Peak reserved GPU memory: "
        f"{peak_memory_reserved_gb:.3f} GB"
    )
    print("=" * 170)
    x_final = x_next
    p_final = x_final[:,0,:,:].squeeze(0)
    u_final = x_final[:,1,:,:].squeeze(0)
    v_final = x_final[:,2,:,:].squeeze(0)

    p_final = normalize_field(p_final, p_stats)
    u_final = normalize_field(u_final, u_stats)
    v_final = normalize_field(v_final, v_stats)

    relative_error_p = np.abs(np.mean(p_final.detach().cpu().numpy()) - np.mean(p_GT.detach().cpu().numpy())) / np.mean(p_GT.detach().cpu().numpy())
    relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
    relative_error_v = torch.norm(v_final - v_GT, 2) / torch.norm(v_GT, 2)

    print(f'true P: {np.mean(p_GT.detach().cpu().numpy())}')
    print(f'predicted P: {np.mean(p_final.detach().cpu().numpy())}')
    print(f'Relative error of p: {relative_error_p}')
    print(f'Relative error of u: {relative_error_u}')
    print(f'Relative error of v: {relative_error_v}')
    print("*"*170)

    p_gt_np, u_gt_np, v_gt_np = p_GT.detach().cpu().numpy(), u_GT.detach().cpu().numpy(), v_GT.detach().cpu().numpy()
    p_pred_np, u_pred_np, v_pred_np = p_final.detach().cpu().numpy(), u_final.detach().cpu().numpy(), v_final.detach().cpu().numpy()



    import matplotlib.pyplot as plt
    plt.imshow(p_gt_np)
    plt.savefig('p_gt.png')
    plt.close()
    plt.imshow(p_pred_np)
    plt.savefig('p_pred.png')
    plt.close()
    plt.imshow(u_gt_np)
    plt.savefig('u_gt.png')
    plt.close()
    plt.imshow(u_pred_np)
    plt.savefig('u_pred.png')
    plt.close()
    plt.imshow(v_gt_np)
    plt.savefig('v_gt.png')
    plt.close()
    plt.imshow(v_pred_np)
    plot_comparison_grid(
        p_gt=p_gt_np,
        u_gt=u_gt_np,
        v_gt=v_gt_np,
        p_pred=p_pred_np,
        u_pred=u_pred_np,
        v_pred=v_pred_np,
        save_dir=save_dir,
        offset=offset,
    )


    # --- Save the metrics ---

    save_and_plot_metrics(
        save_dir=save_dir,
        offset=offset,
        relative_error_p=relative_error_p,
        relative_error_u=relative_error_u.detach().cpu().item(),
        relative_error_v=relative_error_v.detach().cpu().item(),
        TypeProblem=TypeProblem,
    )

