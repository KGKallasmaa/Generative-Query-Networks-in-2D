import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Load dataset
from shepardmetzler import ShepardMetzler
from torch.utils.data import DataLoader
from gqn import GenerativeQueryNetwork, partition

dataset = ShepardMetzler("./our_data_vol2") ## <= Choose your data location
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model parameters onto CPU
state_dict = torch.load("./checkpoint_model_253.pth", map_location="cpu") ## <= Choose your model location

# Initialise new model with the settings of the trained one
model_settings = dict(x_dim=3, v_dim=7, r_dim=512, h_dim=128, z_dim=64, L=8)
model = GenerativeQueryNetwork(**model_settings)

# Load trained parameters, un-dataparallel if needed
if True in ["module" in m for m in list(state_dict().keys())]:
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict())
    model = model.module
else:
    model.load_state_dict(state_dict())

model

# We load a batch of a single image containing a single object seen from 15 different viewpoints.

def deterministic_partition(images, viewpoints, indices):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    _, b, m, *x_dims = images.shape
    _, b, m, *v_dims = viewpoints.shape

    # "Squeeze" the batch dimension
    images = images.view((-1, m, *x_dims))
    viewpoints = viewpoints.view((-1, m, *v_dims))

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    return x, v, x_q, v_q
import random

# Pick a scene to visualise
scene_id = 3

# Load data
x, v = next(iter(loader))
x_, v_ = x.squeeze(0), v.squeeze(0)

# Sample a set of views
n_context = 13 + 1
indices = random.sample([i for i in range(v_.size(1))], n_context)

# Seperate into context and query sets
x_c, v_c, x_q, v_q = deterministic_partition(x, v, indices)

# Visualise context and query images
f, axarr = plt.subplots(1, 15, figsize=(20, 7))
for i, ax in enumerate(axarr.flat):
    # Move channel dimension to end
    ax.imshow(x_[scene_id][i].permute(1, 2, 0))

    if i == indices[-1]:
        ax.set_title("Query", color="magenta")
    elif i in indices[:-1]:
        ax.set_title("Context", color="green")
    else:
        ax.set_title("Unused", color="grey")

    ax.axis("off")

#### Reconstruction
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 7))

x_mu, r, kl = model(x_c[scene_id].unsqueeze(0),
                    v_c[scene_id].unsqueeze(0),
                    x_q[scene_id].unsqueeze(0),
                    v_q[scene_id].unsqueeze(0))

x_mu = x_mu.squeeze(0)
r = r.squeeze(0)

ax1.imshow(x_q[scene_id].data.permute(1, 2, 0))
ax1.set_title("Query image")
ax1.axis("off")

ax2.imshow(x_mu.data.permute(1, 2, 0))
ax2.set_title("Reconstruction")
ax2.axis("off")

ax3.imshow(r.data.view(16, 32))
ax3.set_title("Representation")
ax3.axis("off")

plt.show()

#### Visualising representation
# We might be interested in visualising the representation
# as more context points are introduced.
f, axarr = plt.subplots(1, 7, figsize=(20, 7))

r = torch.zeros(64, 256, 1, 1)

for i, ax in enumerate(axarr.flat):
    phi = model.representation(x_c[:, i], v_c[:, i])
    r += phi
    ax.imshow(r[scene_id].data.view(16, 16))
    ax.axis("off")
    ax.set_title("#Context points: {}".format(i+1))


#### Sample from the prior.
# Create progressively growing context set
batch_size, n_views, c, h, w = x_c.shape
num_samples = 7

f, axarr = plt.subplots(1, num_samples, figsize=(20, 7))
for i, ax in enumerate(axarr.flat):
    x_ = x_c[scene_id][:i+1].view(-1, c, h, w)
    v_ = v_c[scene_id][:i+1].view(-1, 7)

    phi = model.representation(x_, v_)

    r = torch.sum(phi, dim=0)
    x_mu = model.generator.sample((h, w), v_q[scene_id].unsqueeze(0), r)
    ax.imshow(x_mu.squeeze(0).data.permute(1, 2, 0))
    ax.set_title("Context points: {}".format(i))
    ax.axis("off")

#### Mental rotation task
# Change viewpoint yaw
batch_size, n_views, c, h, w = context_x.shape
pi = 3.1415629

x_ = x_c[scene_id].view(-1, c, h, w)
v_ = v_c[scene_id].view(-1, 7)

phi = model.representation(x_, v_)

r = torch.sum(phi, dim=0)

f, axarr = plt.subplots(2, num_samples, figsize=(20, 7))
for i, ax in enumerate(axarr[0].flat):
    v = torch.zeros(7).copy_(v_q[scene_id])

    yaw = (i+1) * (pi/8) - pi/2
    v[3], v[4] = np.cos(yaw), np.sin(yaw)

    x_mu = model.generator.sample((h, w), v.unsqueeze(0), r)
    ax.imshow(x_mu.squeeze(0).data.permute(1, 2, 0))
    ax.set_title(r"Yaw:" + str(i+1) + r"$\frac{\pi}{8} - \frac{\pi}{2}$")
    ax.axis("off")

for i, ax in enumerate(axarr[1].flat):
    v = torch.zeros(7).copy_(v_q[scene_id])

    pitch = (i+1) * (pi/8) - pi/2
    v[5], v[6] = np.cos(pitch), np.sin(pitch)

    x_mu = model.generator.sample((h, w), v.unsqueeze(0), r)
    ax.imshow(x_mu.squeeze(0).data.permute(1, 2, 0))
    ax.set_title(r"Pitch:" + str(i+1) + r"$\frac{\pi}{8} - \frac{\pi}{2}$")
    ax.axis("off")
