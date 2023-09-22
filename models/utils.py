# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import itertools
import random
from matplotlib.pyplot import title
import torch
import cv2
import numpy as np
import torch.distributed as dist
from torch._six import inf
import scipy.optimize
# from pyinstrument import Profiler
from PIL import Image
# profiler = Profiler(interval=0.0001)

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    if epoch%1 == 0:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# def compute_fov(f, xi, width):
#     return 2 * torch.arccos((xi + torch.sqrt(1 + (1 - xi**2) * (width/2/f)**2)) / ((width/2/f)**2 + 1) - xi)

# # compute focal length from field of view and xi
# def compute_focal(fov, xi, width):
#     return width / 2 * (xi + torch.cos(fov/2)) / torch.sin(fov/2)

def distort_image(img, D, shift=(0.0, 0.0)) -> np.ndarray:
    """Distort an image using a fisheye distortion model
    Args:
        img (PIL): the image to distort
        alpha (float): fov angle (radians)
        D (list[float]): a list containing the k1, k2, k3 and k4 parameters
        shift (tuple[float, float]): x and y shift (respectively)
    Returns:
        np.ndarray: the distorted image
    """

    img = img.resize((384, 384), Image.ANTIALIAS)
    img = np.array(img)
    # print(img.shape)
    
    try:
        height, width, _= img.shape
    except:
        img = np.stack((img, img, img), axis=2)
        height, width, _= img.shape
    center = [height//2, width//2]

    # Image coordinates
    map_x, map_y = np.mgrid[0:height, 0:width].astype(np.float32)

    # Center coordinate system
    if height % 2 == 0:
        center[0] -= 0.5
    if width % 2 == 0:
        center[1] -= 0.5

    map_x -= center[0]
    map_y -= center[1]

    # (shift and) convert to polar coordinates
    r = np.sqrt((map_x + shift[0])**2 + (map_y + shift[1])**2)
    theta = (r * (np.pi / 2)) / height

    # Compute fisheye distortion with equidistant projection
    theta_d = theta * (1 + D[0]*theta**2 + D[1]*theta**4 + D[2]*theta**6 + D[3]*theta**8)

    # Scale so that image always fits the original size
    f = map_y.max() / theta_d[int(center[0]), 0]
    r_d = f * theta_d

    # Compute distorted map and rotate
    map_xd = (r_d / r) * map_x + center[0]
    map_yd = (r_d / r) * map_y + center[1]

    # Distort
    distorted_image = cv2.remap(
        img, map_yd, map_xd,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )

    distorted_image = Image.fromarray(distorted_image)
    distorted_image = distorted_image.resize((64, 64), Image.ANTIALIAS)
    rgb_im = distorted_image.convert('RGB')

    return rgb_im

def distort_batch(x, alpha, D, shift=(0.0, 0.0), phi=0.0) :
    """Distort a batch of images (in-place) using a fisheye distortion model (same as distort_image but for a batch of images)
    Args:
        x (torch.Tensor): the batch to distort
        alpha (float): fov angle (radians)
        D (list[float]): a list containing the k1, k2, k3 and k4 parameters
        shift (tuple[float, float]): x and y shift (respectively)
        phi (float): the rotation angle (radians)
    Returns:
        torch.Tensor: the distorted batch
    """
    arr = x.moveaxis(1, -1).numpy()
    for i in range(arr.shape[0]):
        arr[i] = distort_image(arr[i], alpha, D, shift, phi)
    return x

def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def get_sample_locations(alpha, phi, dmin, ds, n_azimuth, n_radius, img_size, subdiv, distort ,  fov=0, focal=0,xi=0,  radius_buffer=0, azimuth_buffer=0):
    # import pdb;pdb.set_trace()
    """Get the sample locations in a given radius and azimuth range
    
    Args:
        alpha (array): width of the azimuth range (radians)
        phi (array): phase shift of the azimuth range  (radians)
        dmin (array): minimum radius of the patch (pixels)
        ds (array): distance between the inner and outer arcs of the patch (pixels)
        n_azimuth (int): number of azimuth samples
        n_radius (int): number of radius samples
        img_size (tuple): the size of the image (width, height)
        radius_buffer (int, optional): radius buffer (pixels). Defaults to 0.
        azimuth_buffer (int, optional): azimuth buffer (radians). Defaults to 0.
    
    Returns:
        tuple[ndarray, ndarray]: lists of x and y coordinates of the sample locations
    """
    #Compute center of the image to shift the samples later
    # import pdb;pdb.set_trace()
    new_f = focal
    rad = lambda x: new_f*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    inverse_rad = lambda r: torch.tan(torch.arcsin(xi*r/(new_f)/torch.sqrt(1 + (r/(new_f))*(r/(new_f)))) + torch.arctan(r/(new_f)))

    center = [img_size[0]/2, img_size[1]/2]
    if img_size[0] % 2 == 0:
        center[0] -= 0.5
    if img_size[1] % 2 == 0:
        center[1] -= 0.5
    # import pdb;pdb.set_trace()
    # Sweep start and end
    r_end = dmin + ds 
    # - radius_buffer
    r_start = dmin 
    # + radius_buffer
    alpha_start = phi 
    B = dmin.shape[1]

    # + azimuth_buffer
    alpha_end = alpha + phi 
    # - azimuth_buffer
    # import pdb;pdb.set_trace()
    # Get the sample locations
    # import pdb;pdb.set_trace()
    # r1 = linspace(r_start, r_end, n_radius)
    if distort == 'spherical':
        radius = linspace(inverse_rad(r_start), inverse_rad(r_end), n_radius)
        radius = rad(radius)
    elif distort  == 'polynomial':
        radius = linspace(r_start, r_end, n_radius)
    # import pdb;pdb.set_trace()
    radius = torch.transpose(radius, 0,1)
    radius = radius.reshape(radius.shape[0]*radius.shape[1], B)
    azimuth = linspace(alpha_start, alpha_end, n_azimuth)
    azimuth = torch.transpose(azimuth, 0,1)
    azimuth = azimuth.flatten()
    azimuth = azimuth.reshape(azimuth.shape[0], 1).repeat_interleave(B, 1)

    
    azimuth = azimuth.reshape(1, azimuth.shape[0], B).repeat_interleave(n_radius, 0)
    radius = radius.reshape(radius.shape[0], 1, B).repeat_interleave(n_azimuth, 1)
    # import pdb;pdb.set_trace()
    radius_mesh = radius.reshape(subdiv[0]*subdiv[1], n_radius, n_azimuth, B)
    # import pdb;pdb.set_trace()
    # d = radius_mesh[0][0][0][0] - radius_mesh[0][1][0][0]
    # eps = np.random.normal(0, d/3)
    # radius_mesh = random.uniform(radius_mesh-d, radius_mesh+d)
    # radius_mesh = radius_mesh + eps
    # import pdb;pdb.set_trace()
    azimuth_mesh = azimuth.reshape(n_radius, subdiv[0]*subdiv[1], n_azimuth, B).transpose(0,1)  
    azimuth_mesh_cos  = torch.cos(azimuth_mesh) 
    azimuth_mesh_sine = torch.sin(azimuth_mesh) 
    x = radius_mesh * azimuth_mesh_cos    # takes time the cosine and multiplication function 
    y = radius_mesh * azimuth_mesh_sine
    # import pdb;pdb.set_trace()
    return x.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1), y.reshape(subdiv[0]*subdiv[1], n_radius*n_azimuth, B).transpose(1, 2).transpose(0,1)


def get_inverse_distortion(num_points, D, max_radius):
    # import pdb;pdb.set_trace()
    dist_func = lambda x: x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (1 + torch.outer(D[0], x**2).flatten() + torch.outer(D[1], x**4).flatten() + torch.outer(D[2], x**6).flatten() +torch.outer(D[3], x**8).flatten())

    theta_max = dist_func(torch.tensor([1]).cuda())
    # import pdb;pdb.set_trace()
    theta = linspace(torch.tensor([0]).cuda(), theta_max, num_points+1).cuda()

    test_radius = torch.linspace(0, 1, 50).cuda()
    test_theta = dist_func(test_radius).reshape(D.shape[1], 50).transpose(1,0)

    radius_list = torch.zeros(num_points*D.shape[1]).reshape(num_points, D.shape[1]).cuda()
    # import pdb;pdb.set_trace()
    for i in range(D.shape[1]):
        for j in range(num_points):
            lower_idx = test_theta[:, i][test_theta[:, i] <= theta[:, i][j]].argmax()
            upper_idx = lower_idx + 1

            x_0, x_1 = test_radius[lower_idx], test_radius[upper_idx]
            y_0, y_1 = test_theta[:, i][lower_idx], test_theta[:, i][upper_idx]

            radius_list[:, i][j] = x_0 + (theta[:, i][j] - y_0) * (x_1 - x_0) / (y_1 - y_0)
    
    # import pdb;pdb.set_trace()
    max_rad = torch.tensor([1]*D.shape[1]).reshape(1, D.shape[1]).cuda()
    return torch.cat((radius_list, max_rad), axis=0)*max_radius, theta_max

def get_inverse_dist_spherical(num_points, xi, fov, new_f):
    # import pdb;pdb.set_trace()
    # xi = torch.tensor(xi).cuda()
    # width = torch.tensor(width).cuda()
    # # focal_length = torch.tensor(focal_length).cuda()
    # fov = compute_fov(focal_length, 0, width)
    # new_xi = xi
    # new_f = compute_focal(fov, new_xi, width)
    # import pdb;pdb.set_trace()
    rad = lambda x: new_f*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    # rad_1 = lambda x: new_f/8*torch.sin(torch.arctan(x))/(xi + torch.cos(torch.arctan(x))) 
    inverse_rad = lambda r: torch.tan(torch.arcsin(xi*r/(new_f)*(1 + (r/(new_f))*(r/(new_f)))) + torch.arctan(r/(new_f)))
#     theta_d_max = inverse_rad(new_f)
    min = inverse_rad(2.0)
    theta_d_max = torch.tan(fov/2).cuda()
    theta_d = linspace(torch.tensor([0]).cuda(), theta_d_max, num_points+1).cuda()
    t1 = inverse_rad(2.0)
    t2 = inverse_rad(4.0)
    # theta_d_num = linspace(torch.tensor([0]).cuda(), theta_d_max, (num_points+1)*8).cuda()
    theta_d_num1 = linspace(t1, t2, 10).cuda()
    r_list = rad(theta_d)   
    # r_lin = rad(theta_d_num)
    # r_d = rad(theta_d_num1)
    # import pdb;pdb.set_trace()
    return r_list, theta_d_max

def get_sample_params_from_subdiv(subdiv, n_radius, n_azimuth, distortion_model, img_size, D=torch.tensor(np.array([0.5, 0.5, 0.5, 0.5]).reshape(4,1)).cuda(), radius_buffer=0, azimuth_buffer=0):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    # import pdb;pdb.set_trace()
    max_radius = min(img_size)/2
    width = img_size[1]
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        # import pdb;pdb.set_trace()
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
    elif distortion_model == 'polynomial':
        # import pdb;pdb.set_trace()
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, max_radius)
    # import pdb;pdb.set_trace()
    # D_min = np.array(dmin_list)  ## del
    D_s = torch.diff(D_min, axis = 0)
    # D_s = np.array(ds_list)
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]
    # import pdb;pdb.set_trace()

    D_min = D_min[:-1].reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    D_s = D_s.reshape(1, subdiv[0], D.shape[1]).repeat_interleave(subdiv[1], 0).reshape(subdiv[0]*subdiv[1], D.shape[1])
    phi_start = 0
    phi_end = 2*torch.tensor(np.pi)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step)
    p = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0)
    phi = p.transpose(1,0).flatten().cuda()
    alpha = alpha.repeat_interleave(subdiv[0]*subdiv[1])
    # Generate parameters for each patch
    # import pdb;pdb.set_trace()
    if distortion_model == 'spherical':
        params = {
            'alpha': alpha, "phi": phi, "dmin": D_min, "ds": D_s, "n_azimuth": n_azimuth, "n_radius": n_radius,
            "img_size": img_size, "radius_buffer": radius_buffer, "azimuth_buffer": azimuth_buffer, "subdiv" : subdiv, "fov": fov, "xi": xi, "focal" : f, "distort" : distortion_model,
        }
    elif distortion_model == 'polynomial':
        params = {
            'alpha': alpha, "phi": phi, "dmin": D_min, "ds": D_s, "n_azimuth": n_azimuth, "n_radius": n_radius,
            "img_size": img_size, "radius_buffer": radius_buffer, "azimuth_buffer": azimuth_buffer, "subdiv" : subdiv, "distort" : distortion_model,
        }
    # import pdb;pdb.set_trace()

    return params, D_s.reshape(subdiv[1], subdiv[0], D.shape[1]).T, theta_max



# def get_optimal_buffers(subdiv, n_radius, n_azimuth, img_size):
#     """Get the optimal radius and azimuth buffers for a given subdivision

#     Args:
#         subdiv (int or tuple[int, int]): the number of subdivisions for which we need to create the samples.
#                                          If specified as a tuple, the format is (radius_subdiv, azimuth_subdiv)
#         n_radius (int): number of radius samples
#         n_azimuth (int): number of azimuth samples
#         img_size (tuple): the size of the image

#     Returns:
#         tuple[int, int]: the optimal radius and azimuth buffers
#     """

#     # Get the optimal buffers
#     if isinstance(subdiv, int):
#         radius_buffer = img_size[0] / (2**(subdiv+1)*n_radius)
#         azimuth_buffer = 2*np.pi / (2**(subdiv+2)*n_azimuth)
#     elif isinstance(subdiv, tuple) and len(subdiv) == 2:
#         radius_buffer = img_size[0] / (radius_subdiv*n_radius*2*2)
#         azimuth_buffer = 2*np.pi / (azimuth_subdiv*n_azimuth*2)
#     else:
#         raise ValueError("Invalid subdivision")
   
#     return radius_buffer, azimuth_buffer



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


if __name__=='__main__':
    profiler.start()
    # import matplotlib.pyplot as plt
    # _, ax = plt.subplots(figsize=(8, 8))
    # ax.set_title("Sampling locations")
    # colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

    # subdiv = 3
    radius_subdiv = 2
    azimuth_subdiv = 8
    subdiv = (radius_subdiv, azimuth_subdiv)
    # subdiv = 3
    n_radius = 8
    n_azimuth = 8
    img_size = (64, 64)
    # radius_buffer, azimuth_buffer = get_optimal_buffers(subdiv, n_radius, n_azimuth, img_size)
    radius_buffer = azimuth_buffer = 0

    D = torch.tensor(np.array([0.0, 0.0, 0.0, 0.0]).reshape(1,4).transpose(1,0)).cuda()
    # import pdb;pdb.set_trace()

    params, D_s = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        n_radius=n_radius,
        D = D,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer
    )

    import pdb;pdb.set_trace()

    sample_locations = get_sample_locations(**params)
    profiler.stop()
    profiler.print()
    print(sample_locations[0].shape)