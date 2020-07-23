import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    # DECLARATION ORDER MUST MATCH C++ IMPLEMENTATION
    BASE = 288
    self.white_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.black_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.fc0 = nn.Linear(2*BASE, 32)
    self.fc1 = nn.Linear(32, 32)
    self.fc_mu = nn.Linear(32, 1)
    self.fc_sigma = nn.Linear(32, 1)

  def forward(self, pov, white, black):
    w_ = self.white_affine(util.half_kp(white, black))
    b_ = self.black_affine(util.half_kp(black, white))
    base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=-1))
    x = F.relu(self.fc0(base))
    x = F.relu(self.fc1(x))
    mu = self.fc_mu(x)
    sigma = self.fc_sigma(x).exp()
    return mu, sigma

  def to_binary_file(self, path):
    joined = np.array([])
    for p in self.parameters():
      print(p.size())
      joined = np.concatenate((joined, p.data.cpu().t().flatten().numpy()))
    print(joined.shape)
    joined.astype('float32').tofile(path)


def variational_loss_fn(outcome, score, pred, lambda_):
  logit, _ = util.cp_conversion(score)
  u, s = pred

  # get log-likelihood of logit-normal distribution
  print(s)
  log_density = s.log() + 0.5 * ((logit - u) / s) ** 2

  # reparameterization trick to sample from logit-normal
  t = outcome
  sample = u + s * torch.randn_like(score)
  cross_entropy = -(t * F.logsigmoid(sample) + (1.0 - t) * F.logsigmoid(-sample))

  # sum losses
  return (cross_entropy + log_density).mean()
