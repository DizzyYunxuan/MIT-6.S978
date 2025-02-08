import matplotlib.pyplot as plt
import numpy as np


timesteps = 1000
beta1 = 0.1
beta2 = 50.0
dt = 1.0 / timesteps
means = np.array([0.5, -0.5])
stds = np.array([0.02, 0.02])
weights = np.array([0.5, 0.5])
weights /= np.sum(weights)
n_samples = 100
x_min = -3
x_max = 3
x_grid = np.linspace(x_min, x_max, num=200)

def get_beta_t(t):
    ratio = float(t) / timesteps
    return ratio * beta2 + (1 - ratio) * beta1

def f(x, t):
    beta_t = get_beta_t(t)
    return -0.5 * beta_t * x

def g(t):
    beta_t = get_beta_t(t)
    return np.sqrt(beta_t)

def gaussian_pdf(x, mean, std):
    return (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def mixture_pdf(x):
    pdf = np.zeros_like(x)
    for i in range(len(means)):
        pdf += weights[i] * gaussian_pdf(x, means[i], stds[i])
    return pdf

def sample_mixture_gaussian(n_samples):
    components = np.random.choice(len(means), size=n_samples, p=weights / np.sum(weights))
    samples = np.random.normal(loc=means[components], scale=stds[components])
    return samples

def p_xt(x_t, t):
    p_xt_val = np.zeros_like(x_t)
    ##################
    ### Problem 3(a): p(x(t))
    ##################
    ##################
    sigma_xt_x0 = np.sqrt(beta1 ** 2 * (beta2 / beta1) ** (2 * t / timesteps))
    p_x0 = mixture_pdf(x_t)
    # p_xt_x0 = gaussian_pdf(x_t, p_x0, sigma)

    # p_xt_val = p_x0 * p_xt_x0

    p_xt_val = np.zeros_like(x_t)
    for i in range(len(means)):
        # mu_it = (means[i] * sigma_xt_x0 + x_t) / (sigma_xt_x0 + stds[i] ** 2)
        # sigma_it = sigma_xt_x0 + stds[i] ** 2
        p_xt_val += weights[i] * (p_x0 + gaussian_pdf(x_t, 0, sigma_xt_x0))
    
    return p_xt_val

def grad_log_p_xt(x_t, t):
    ##################
    ### Problem 3(b): \nabla_x(t) \log p(x(t))
    ##################
    ##################

    # grad = np.zeros_like(x_t)
    beta_t = beta1 + (t / timesteps) * (beta2 - beta1)

    grad = -1.0 * (x_t - 0.0) / beta_t ** 2


    return grad

def forward_sde(timesteps, n_samples, dt):
    x = np.zeros((timesteps, n_samples))

    x_pdf = np.zeros((timesteps, x_grid.shape[0]))

    x0 = sample_mixture_gaussian(n_samples)
    x0_pdf = mixture_pdf(x_grid)
    x[0] = x0
    x_pdf[0] = x0_pdf

    for t in range(1, timesteps):
        noise = np.random.normal(0, 1, size=n_samples)
        x[t] = x[t-1] + f(x[t-1], t) * dt + g(t) * noise * np.sqrt(dt)
        x_pdf[t] = p_xt(x_grid, t)

    return x, x_pdf

def backward_sde(timesteps, n_samples, dt):
    x = np.zeros((timesteps, n_samples))
    xT = np.random.normal(0, 1, size=n_samples)
    x[-1] = xT

    for t in range(timesteps - 1, 0, -1):
        noise = np.random.normal(0, 1, size=n_samples)
        delta_x = (f(x[t], t) - g(t) ** 2 * grad_log_p_xt(x[t], t)) * dt + g(t) * noise * np.sqrt(dt)
        x[t-1] = x[t] + delta_x

    return x

forward_x, forward_x_pdf = forward_sde(timesteps, n_samples, dt)
backward_data = backward_sde(timesteps, n_samples, dt)

fig, axes = plt.subplots(1, 2, figsize=(36, 12))

for i in range(n_samples):
    axes[0].plot(forward_x[:, i], lw=1)

time = np.arange(timesteps)
X, Y = np.meshgrid(time, x_grid)
pcm = axes[0].pcolormesh(X, Y, forward_x_pdf.T,
                         cmap='viridis', shading='auto', vmin=0.0, vmax=forward_x_pdf.max())
axes[0].set_title('Forward SDE')
axes[0].set_xlabel('Timesteps')
axes[0].set_ylabel('x')
axes[0].set_ylim([x_min, x_max])

for i in range(n_samples):
    axes[1].plot(backward_data[::-1][:, i], lw=1)

X, Y = np.meshgrid(time, x_grid)
pcm = axes[1].pcolormesh(X, Y, forward_x_pdf[::-1].T,
                         cmap='viridis', shading='auto', vmin=0.0, vmax=forward_x_pdf.max())
axes[1].set_title('Reverse SDE')
axes[1].set_xlabel('Timesteps')
axes[1].set_ylabel('x')

plt.tight_layout()
plt.show()

