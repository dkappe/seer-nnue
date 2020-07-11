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
    
    self.val_fc0 = nn.Linear(2*BASE, 32)
    self.val_fc1 = nn.Linear(32, 32)
    self.val_fc2 = nn.Linear(32, 1)
    
    self.act_fc0 = nn.Linear(2*BASE, 144)
    self.act_fc1 = nn.Linear(144, util.move_numel())

    self.val_skip = nn.Linear(2*BASE, 1)
    self.act_skip = nn.Linear(2*BASE, util.move_numel())

  def forward(self, pov, white, black):
    w_256 = self.white_affine(util.half_kp(white, black))
    b_256 = self.black_affine(util.half_kp(black, white))
    base = F.relu(pov * torch.cat([w_256, b_256], dim=1) + (1.0 - pov) * torch.cat([b_256, w_256], dim=1))

    val = F.relu(self.val_fc0(base))
    val = F.relu(self.val_fc1(val))
    val = self.val_fc2(val) + self.val_skip(base)
    
    act = F.relu(self.act_fc0(base))
    act = self.act_fc1(act) + self.act_skip(base)
    act = act.reshape(act.size(0), *util.move_size())
    
    return val, act

  def to_binary_file(self, path):
    joined = np.array([])
    for p in self.parameters():
      print(p.size())
      joined = np.concatenate((joined, p.data.cpu().t().flatten().numpy()))
    print(joined.shape)
    joined.astype('float32').tofile(path)


def value_loss_fn(outcome, score, pred, lambda_):
  q = pred
  t = outcome
  p = util.cp_conversion(score)
  teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
  outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
  result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
  return result.sum()


def action_loss_fn(move_mask, actual_move, pred):
  EPSILON = 1e-12
  mask = (move_mask + EPSILON).clamp(0.0, 1.0).log()
  pred = F.log_softmax(pred - pred.min(dim=-1, keepdim=True)[0] + mask, dim=-1)
  return -(pred * actual_move).sum()

