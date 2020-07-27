import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    # DECLARATION ORDER MUST MATCH C++ IMPLEMENTATION
    BASE = 320
    self.white_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.black_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.fc0 = nn.Linear(2*BASE, 32)
    self.drop0 = nn.Dropout(p=0.2)
    self.fc1 = nn.Linear(32, 32)
    self.drop1 = nn.Dropout(p=0.1)
    self.fc_mu = nn.Linear(32, 1)
    self.fc_sigma = nn.Linear(32, 1)

  def clean_encode(self, base):
    x = F.relu(self.fc0(base))
    x = F.relu(self.fc1(x))
    return x

  def noisy_encode(self, base):
    x = self.drop0(base)
    x = F.relu(self.fc0(x))
    x = self.drop1(x)
    x = F.relu(self.fc1(x))
    return x

  def forward(self, pov, white, black):
    w_ = self.white_affine(util.half_kp(white, black))
    b_ = self.black_affine(util.half_kp(black, white))
    base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=-1))
    x_noisy = self.noisy_encode(base)
    x_clean = self.clean_encode(base)
    mu = self.fc_mu(x_noisy)
    sigma = self.fc_sigma(x_clean).exp()
    return mu, sigma

  def to_binary_file(self, path):
    joined = np.array([])
    for p in self.parameters():
      print(p.size())
      joined = np.concatenate((joined, p.data.cpu().t().flatten().numpy()))
    print(joined.shape)
    joined.astype('float32').tofile(path)


def variational_loss_fn(outcome, score, pred):
  logit, _ = util.cp_conversion(score)
  u, s = pred
  #get negative log likelihood for normal distribution
  log_density = s.log() + 0.5 * ((logit - u) / s) ** 2
  return log_density.sum()


