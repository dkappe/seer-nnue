from os import path
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import config as C
import util
import dataset
import nnue_bin_dataset
import model

def train_step(M, sample, opt, queue, max_queue_size, lambda_, report=False):
  pov, white, black, outcome, score = sample
  pred = M(pov, white, black)
  loss = model.variational_loss_fn(outcome, score, pred, lambda_)
  if report:
    print(loss.item())
  loss.backward()
  if(len(queue) >= max_queue_size):
    queue.pop(0)
  queue.append(loss.item())
  opt.step()
  M.zero_grad()


def main():
  config = C.Config('config.yaml')

  M = model.NNUE().to(config.device)

  if (path.exists(config.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(config.model_save_path))

  if config.use_binary_data:
    # Using binary data generated by nodchip's nnue stockfish fork
    data = nnue_bin_dataset.NNUEBinData(config)
  else:
    # Use memmaped numpy arrays generated by generate_data script
    data = dataset.Data(config)

  opt = optim.Adadelta(M.parameters(), lr=config.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, len(data) // config.epoch_length, gamma=0.5)

  loss_history = []
  queue = []
  
  for epoch in range(1, config.epochs + 1):
    M.to_binary_file(config.bin_model_save_path)
    for i in range(config.epoch_length):
      # update visual data
      if (i % config.test_rate) == 0 and i != 0:
        loss_history.append(sum(queue) / len(queue))
        plt.clf()
        plt.plot(loss_history)
        plt.savefig('{}/loss_graph.png'.format(config.visual_directory), bbox_inches='tight')
      
      sample = data.sample_batch()
      train_step(M, sample, opt, queue, max_queue_size=config.max_queue_size, lambda_=config.lambda_, report=(0 == i % config.report_rate))

    torch.save(M.state_dict(), config.model_save_path)
    scheduler.step()


if __name__ == '__main__':
  main()
