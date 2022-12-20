### Utility modules for diffusion

## Todos / Questions:
# 0. Add switched handling for what the diffusion net predicts: (1) noise (2) u_tilde (3) x_0
# 1. [done] Add a switch in calculate_hybrid_loss for (1) MSE over noise (2) MSE over means (3) MSE over x_0
# 2. [done] clamping pred_x_0 in [-1, 1] when using the loss as MSE over x_start - we should ensure that both true x_0 and predicted_x_0 are in range [-1, 1]
# 3. [done] incorporate dynamic thresholding in sampling (idea from IMAGEN paper)
# 4. add handling for inputs with different number of dimensions (not just images with 3 dimensions - h,w,c)

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def linear_noise_schedule(beta_min, beta_max, max_time_steps):
    return torch.linspace(beta_min, beta_max, max_time_steps)

# the function used to calculate cosine factor used in cosine noise schedule
def cosine_func(t, max_time_steps, s=0.008):
    return torch.pow( torch.cos( (((t/max_time_steps)+s) / (1+s)) * torch.tensor(torch.pi/2) ), 2)

def cosine_noise_schedule(max_time_steps):
    betas = []
    # initial values
    f_0 = cosine_func(0, max_time_steps)
    alpha_hat_prev = 1.
    # iterate
    for t in range(1, max_time_steps+1):
        f_t = cosine_func(t, max_time_steps)
        alpha_hat = f_t / f_0
        beta = 1 - (alpha_hat/alpha_hat_prev)
        beta = torch.clamp(beta, min=0., max=0.999)
        betas.append(beta)
        alpha_hat_prev = alpha_hat
    return torch.stack(betas, dim=0)

# utility function to increase alphas_hat dimensions to match input dimensions (enables handling of inputs of different dimensions)
def expand_alphas_hat_dims(alphas_hat, x):
    while len(alphas_hat.shape) < len(x.shape):
        alphas_hat = alphas_hat.unsqueeze(-1)
    return alphas_hat

# OpenAi implementation for cosine schedule
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    betas = np.array(betas)
    betas = torch.from_numpy(betas).float()
    return betas

# function to perform dynamic thresholding on predicted x_0
def dynamic_thresholding(pred_x_0, threshold=.95):
    maxval = torch.max(pred_x_0)
    maxval = threshold * maxval
    if maxval > 1:
        pred_x_0 = pred_x_0.clamp(-maxval, maxval)
        pred_x_0 = pred_x_0 / maxval
    return pred_x_0


## handy functions to convert between the four quantities (x_0, x_t, eps, u_tilde) involved in the diffusion process
## all the functions are obtained using eq(4) and eq(7) from the DDPM paper

def x_t_from_x_0_and_eps(x_0, eps, alphas_hat, t):
    x_t = torch.sqrt(alphas_hat[t]) * x_0 + torch.sqrt(1 - alphas_hat[t]) * eps
    return x_t

def x_0_from_eps_and_x_t(x_t, eps, alphas_hat, t):
    pred_x_0 = ( x_t - torch.sqrt(1 - alphas_hat[t]) * eps ) / torch.sqrt(alphas_hat[t])
    return pred_x_0

def eps_from_x_0_and_x_t(x_0, x_t, alphas_hat, t):
    pred_eps = ( x_t - torch.sqrt(alphas_hat[t]) * x_0 ) / torch.sqrt(1 - alphas_hat[t])
    return pred_eps

def u_tilde_from_x_0_and_x_t(x_0, x_t, alphas_hat, t, t_minus1):
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    alpha = alpha_hat / alpha_hat_prev
    beta = 1 - alpha
    u_tilde = ( torch.sqrt(alpha_hat_prev) * beta * x_0 + torch.sqrt(alpha) * (1 - alpha_hat_prev) * x_t ) / (1 - alpha_hat)
    return u_tilde

def x_0_from_u_tilde_and_x_t(u_tilde, x_t, alphas_hat, t, t_minus1):
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    alpha = alpha_hat / alpha_hat_prev
    beta = 1 - alpha
    pred_x_0 = ( u_tilde * (1 - alpha_hat) -  torch.sqrt(alpha) * (1 - alpha_hat_prev) * x_t ) / ( torch.sqrt(alpha_hat_prev) * beta )
    return pred_x_0

def u_tilde_from_eps_and_x_t(x_t, eps, alphas_hat, t, t_minus1):
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    alpha = alpha_hat / alpha_hat_prev
    beta = 1 - alpha
    u_tilde = ( x_t - ((beta * eps)/torch.sqrt(1 - alpha_hat)) ) / torch.sqrt(alpha)
    return u_tilde

def eps_from_u_tilde_and_x_t(x_t, u_tilde, alphas_hat, t, t_minus1):
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    alpha = alpha_hat / alpha_hat_prev
    beta = 1 - alpha
    eps = ( (u_tilde * torch.sqrt(alpha) - x_t) * (-torch.sqrt(1 - alpha_hat)) ) / beta
    return eps


# def noise_img(img_ori, alphas_hat, t, device):
#     sqrt_alpha_hat = torch.sqrt(alphas_hat[t])[:, None, None, None]
#     sqrt_one_minus_alpha_hat = torch.sqrt(1 - alphas_hat[t])[:, None, None, None]
#     eps = torch.randn_like(img_ori)
#     noised_img = ( sqrt_alpha_hat * img_ori ) + ( sqrt_one_minus_alpha_hat * eps )
#     return noised_img, eps

def noise_img(img_ori, alphas_hat, t):
    eps = torch.randn_like(img_ori)
    noised_img = x_t_from_x_0_and_eps(img_ori, eps, alphas_hat, t)
    return noised_img, eps


# function to calculate KL div between two gaussians - TODO: check dims for diagonal variance
# used to calculate L_t = KL(q_posterior, p)
def kl_normal(mean_q, logvar_q, mean_p, logvar_p):
    # stop gradient on means
    mean_q, mean_p = mean_q.detach(), mean_p.detach()
    return 0.5 * ( -1.0 + logvar_p - logvar_q + torch.exp(logvar_q - logvar_p) + ((mean_q - mean_p)**2) * torch.exp(-logvar_p) )

### functions to calculate L_0 = -log p(x_0 | x_1) - borrowed from OpenAi implementation of improved DDPM

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

# utility function to take mean of a tensor across all dimensions except the first (batch) dimension
def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.shape))))

# function to calculate mean and variance of q_posterior: q(x_t-1 | x_t,x_0)
def q_posterior_mean_variance(x_0, x_t, t, t_minus1, alphas_hat):
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    # its necessary to re-calculate beta and alpha from alphas_hat:
    alpha = alpha_hat / alpha_hat_prev
    beta = 1 - alpha
    mean = ( torch.sqrt(alpha_hat_prev) * beta * x_0 + torch.sqrt(alpha) * (1 - alpha_hat_prev) * x_t ) / (1 - alpha_hat)
    var = ( (1 - alpha_hat_prev) * beta ) / (1 - alpha_hat)
    logvar = torch.log(var)
    return mean, logvar

# function to calculate mean and variance of p(x_t-1 | x_t)
# we have 3 boolean switches for the net's prediction to be one of the 3 alternatives:
# 1. noise (default)
# 2. mean
# 3. x_0
def p_mean_variance(net, x_t, t, t_minus1, enc_out, alphas_hat, pred_mean, pred_x_start):
    out = net(x_t, t, enc_out) # the net predicts the concatenated [noise, frac]
    x_channels = x_t.shape[1]

    # boolean switch for nature of prediction from net
    if pred_x_start:
        pred_x_0, frac = torch.split(out, x_channels, dim=1)
        mean = u_tilde_from_x_0_and_x_t(pred_x_0, x_t, alphas_hat, t, t_minus1)
        pred_noise = eps_from_x_0_and_x_t(pred_x_0, x_t, alphas_hat, t)
    elif pred_mean:
        mean, frac = torch.split(out, x_channels, dim=1)
        pred_x_0 = x_0_from_u_tilde_and_x_t(mean, x_t, alphas_hat, t, t_minus1)
        pred_noise = eps_from_u_tilde_and_x_t(x_t, mean, alphas_hat, t, t_minus1)
    else: # pred noise (default)
        pred_noise, frac = torch.split(out, x_channels, dim=1)
        mean = u_tilde_from_eps_and_x_t(x_t, pred_noise, alphas_hat, t, t_minus1)
        pred_x_0 = x_0_from_eps_and_x_t(x_t, pred_noise, alphas_hat, t)

    # shift frac values to be in [0, 1] from [-1, 1]
    frac = (frac + 1) * 0.5
    # calculate log variance using frac interpolatiion between min_log_var (beta_tilde) and max_log_var (beta)
    alpha_hat = alphas_hat[t]
    alpha_hat_prev = alphas_hat[t_minus1]
    # so its necessary to re-calculate beta from alphas_hat:
    beta = 1 - (alpha_hat / alpha_hat_prev)
    alpha = 1 - beta
    beta_tilde = ( (1 - alpha_hat_prev) * beta ) / (1 - alpha_hat)
    max_logvar = torch.log(beta)
    min_logvar = torch.log(beta_tilde)
    logvar = frac * max_logvar + (1 - frac) * min_logvar

    return mean, logvar, pred_noise, pred_x_0

# function to calculate hybrid loss: L_hybrid = L_simple + lambda * L_vlb
# we have boolean switches for the loss to be one of the 3 alternatives:
# 1. mse over predicted_noise and true noise (default)
# 2. mse over learnt reverse process mean (p_mean) and surrogate reverse process mean (q_mean)
# 3. mse over predicted x_start and true x_start (predicted x_start can be calculated from predicted_noise and x_t)
# we have 3 boolean switches for the net's prediction to be one of the 3 alternatives:
# 1. noise (default)
# 2. mean
# 3. x_0
def calculate_hybrid_loss(net, x_0, t, enc_out, L_lambda, alphas_hat, mse_over_x_start=False, mse_over_mean=False, pred_mean=False, pred_x_start=False):
    alphas_hat = expand_alphas_hat_dims(alphas_hat, x_0)
    x_t, true_noise = noise_img(x_0, alphas_hat, t)
    q_mean, q_logvar = q_posterior_mean_variance(x_0, x_t, t, t-1, alphas_hat)
    p_mean, p_logvar, pred_noise, pred_x_0 = p_mean_variance(net, x_t, t, t-1, enc_out, alphas_hat, pred_mean, pred_x_start)
    # for t == 1:
    p_log_scale = 0.5 * p_logvar
    L_vlb_0 = -1 * discretized_gaussian_log_likelihood(x_0, p_mean, p_log_scale)
    L_vlb_0 = L_vlb_0 / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_0 = mean_flat(L_vlb_0) # take mean across all dims except batch_dim # shape: [batch_size]

    if mse_over_x_start:
        L_simple_0 = torch.pow(pred_x_0 - x_0, 2)
    elif mse_over_mean:
        L_simple_0 = torch.pow(q_mean - p_mean, 2)
    else: # mse over noise
        L_simple_0 = torch.pow(pred_noise - true_noise, 2)
    L_simple_0 = mean_flat(L_simple_0)
    L_hybrid_0 = L_vlb_0
    # L_hybrid_0 = L_simple_0

    # for t > 1:
    if mse_over_x_start:
        L_simple = torch.pow(pred_x_0 - x_0, 2)
    elif mse_over_mean:
        L_simple = torch.pow(q_mean - p_mean, 2)
    else:
        L_simple = torch.pow(pred_noise - true_noise, 2)
    L_simple = mean_flat(L_simple) # take mean across all dims except batch_dim # shape: [batch_size]
    L_vlb_t = kl_normal(q_mean, q_logvar, p_mean, p_logvar)
    L_vlb_t = L_vlb_t / torch.log(torch.tensor(2.0)) # convert loss from nats to bits
    L_vlb_t = mean_flat(L_vlb_t) # take mean across all dims except batch_dim # shape: [batch_size]
    L_hybrid_t = L_simple + L_lambda * L_vlb_t

    # populate final loss vector according to t values
    L_hybrid = torch.where((t == 1), L_hybrid_0, L_hybrid_t) # shape: [batch_size]
    L_hybrid = L_hybrid.mean() # final loss scalar
    return L_hybrid

# function to sample x_t-1 ~ p(x_t-1 | x_t)
def p_sample_CFG(i, net, x_t, t, t_minus1, enc_label, alphas_hat, guidance_strength, cfg_flag, pred_mean, pred_x_start):
    mean_cond, logvar_cond, pred_noise_cond, pred_x_0 = p_mean_variance(net, x_t, t, t_minus1, enc_label, alphas_hat, pred_mean, pred_x_start)
    if cfg_flag:
        mean_uncond, logvar_uncond, pred_noise_uncond, pred_x_0 = p_mean_variance(net, x_t, t, t_minus1, None, alphas_hat, pred_mean, pred_x_start)
        # calculated interpolated mean (weighted by guidance strength)
        mean_interpolated = mean_cond + guidance_strength * ( mean_cond - mean_uncond )
    else:
        mean_interpolated = mean_cond
    # sample
    eps = torch.randn_like(x_t)
    if i == 1:
        eps = eps * 0
    x_t_minus1 = mean_interpolated + torch.exp(0.5 * logvar_cond) * eps
    return x_t_minus1

# function to sample x_t-1 ~ p(x_t-1 | x_t) - incorporates dynamic thresholding
def p_sample_CFG_with_dynamic_thresholding(i, net, x_t, t, t_minus1, enc_label, alphas_hat, guidance_strength, cfg_flag, pred_mean, pred_x_start):
    mean_cond, logvar_cond, pred_noise_cond, pred_x_0 = p_mean_variance(net, x_t, t, t_minus1, enc_label, alphas_hat, pred_mean, pred_x_start)
    if cfg_flag:
        mean_uncond, logvar_uncond, pred_noise_uncond, pred_x_0 = p_mean_variance(net, x_t, t, t_minus1, None, alphas_hat, pred_mean, pred_x_start)
        # calculated interpolated mean (weighted by guidance strength)
        mean_interpolated = mean_cond + guidance_strength * ( mean_cond - mean_uncond )
    else:
        mean_interpolated = mean_cond
    # re-calculate mean after incorporating dynamic thresholding
    pred_x_0 = x_0_from_u_tilde_and_x_t(mean_interpolated, x_t, alphas_hat, t, t_minus1)
    pred_x_0 = dynamic_thresholding(pred_x_0)
    mean_interpolated = u_tilde_from_x_0_and_x_t(pred_x_0, x_t, alphas_hat, t, t_minus1)
    # sample
    eps = torch.randn_like(x_t)
    if i == 1:
        eps = eps * 0
    x_t_minus1 = mean_interpolated + torch.exp(0.5 * logvar_cond) * eps
    return x_t_minus1

# strided sampling (with classifier free guidance based interpolation)
def sample_strided_CFG(net, alphas_hat, guidance_strength, max_time_steps, subseq_steps, x_shape, device, n, enc_label, cfg_flag=True, pred_mean=False, pred_x_start=False, progress_bar=True):
    x_shape = list(x_shape)
    x_shape = [n] + x_shape
    net.eval()
    with torch.no_grad():
        subseq = torch.linspace(0, max_time_steps-1, subseq_steps, dtype=torch.int)
        x = torch.randn(x_shape).to(device)
        alphas_hat = expand_alphas_hat_dims(alphas_hat, x)
        if progress_bar:
            pb = tqdm( reversed(range(1, subseq_steps)), position=0 )
        else:
            pb = reversed(range(1, subseq_steps))
        for i in pb:
            tau = (torch.ones(n) * subseq[i]).long().to(device)
            tau_minus1 = (torch.ones(n) * subseq[i-1]).long().to(device)
            # x = p_sample_CFG(i, net, x, tau, tau_minus1, enc_label, alphas_hat, guidance_strength, pred_mean, pred_x_start)
            x = p_sample_CFG_with_dynamic_thresholding(i, net, x, tau, tau_minus1, enc_label, alphas_hat, guidance_strength, cfg_flag, pred_mean, pred_x_start)
    net.train()
    return x
