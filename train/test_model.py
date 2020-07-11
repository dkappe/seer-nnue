from os import path
import torch
import chess

import config as C
import util
import model

config = C.Config('config.yaml')

M = model.NNUE().to('cpu')

if (path.exists(config.model_save_path)):
  print('Loading model ... ')
  M.load_state_dict(torch.load(config.model_save_path, map_location='cpu'))

num_parameters = sum(map(lambda x: torch.numel(x), M.parameters()))

print(num_parameters)

M.cpu()

while True:
  bd = chess.Board(input("fen: "))
  w, b = util.to_tensors(bd)
  #print('white_half_kp_indices: ', util.half_kp(w.unsqueeze(0), b.unsqueeze(0)).flatten().nonzero())
  #print('black half_kp_indices: ', util.half_kp(b.unsqueeze(0), w.unsqueeze(0)).flatten().nonzero())
  white, black = util.to_tensors(bd);
  val, act = M(torch.tensor([bd.turn]).float(), white.unsqueeze(0).float(), black.unsqueeze(0).float())
  print(val)
  print(act.flatten())
  act = torch.nn.functional.softmax(act.detach(), dim=-1)
  act = act.reshape(*util.move_size())
  s = 0.0
  ls = []
  for mv in bd.legal_moves:
    val = act[0, mv.from_square] * act[1, mv.to_square]
    s += val
    ls.append((mv, val))
  ls.sort(reverse=True, key=lambda x: x[1])
  for mv, val in ls:
    print(mv.uci(), val / s)
