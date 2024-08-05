# Adapted from https://github.com/lmnt-com/wavegrad under the Apache-2.0 license.

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed
from params import params
import os, shutil


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  print('Dumping hyperparameter file...')
  os.makedirs(args.model_dir, exist_ok=True)
  shutil.copy('params.py', os.path.join(args.model_dir, 'params_saved.py'))
  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a WaveGrad model')
  parser.add_argument('--model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('--data_dirs', nargs='+',
      help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--training_files', nargs='+', default=None,
      help='space separated list of files containing the list of wav samples used for training')
  parser.add_argument('--validation_files', nargs='+', default=None,
      help='space separated list of files containing the list of wav samples used for validation')
  parser.add_argument('--checkpoint_interval', default=None, type=int)
  parser.add_argument('--summary_interval', default=100, type=int)
  parser.add_argument('--validation_interval', default=1000, type=int)
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  main(parser.parse_args())
