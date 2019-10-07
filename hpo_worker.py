__author__ = 'gchlebus'

import time
import os
from worker import KerasWorker


def _parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='Starts one HPO worker. It assumes, that the HPO server is already '
                                               'running.')
  parser.add_argument('output', type=str, help='Output directory (should be the same as specified in the HPO server).')
  parser.add_argument('--interface', '-i', type=str, default="127.0.0.1",
                      help='Communication interface (default:  %(default)s).')
  parser.add_argument('--port', '-p', type=int, default=57945,
                      help='Server port (default:  %(default)d).')
  parser.add_argument('--run-id', type=str, default="0",
                      help='A unique run id for this optimization run (default:  %(default)s)')
  parser.add_argument('--gpu', type=str, default="0",
                      help='GPU to be used')
  parser.add_argument('--data-dir', type=str, default="/data/anneke/prostate-data/preprocessed/train/")
  parser.add_argument('--array-dir', type=str, default="/data/anneke/prostate-data/whole-prostate-arrays/")
  return parser.parse_args()


if __name__ == "__main__":

  args = _parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  time.sleep(5)

  w = KerasWorker(run_id=args.run_id, host=args.interface, out_directory=args.output,
                  nameserver=args.interface, nameserver_port=args.port, data_dir=args.data_dir,
                  array_dir=args.array_dir)
  w.run(background=False)
