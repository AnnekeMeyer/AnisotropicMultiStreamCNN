__author__ = 'gchlebus'

import os
import shutil
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from utils import makedir

CONFIGSPACE_FILENAME = "configspace.py"
OUTPUT_FILENAME = "output.txt"


def _exec(filePath, requiredFunction):
  try:
    with open(filePath, "r") as f:
      exp = f.read()
      exec(exp)
      assert requiredFunction in locals()
      return locals()[requiredFunction]
  except (FileNotFoundError, SyntaxError, AssertionError) as e:
    raise RuntimeError(e)


def copy_file(src, dst):
  makedir(os.path.dirname(dst))
  shutil.copyfile(src, dst)


def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='Starts the HPO server. The HPO server should be started before HPO '
                                               'workers.')
  parser.add_argument('configspace', type=str, help="Path to the config space definition.")
  parser.add_argument('output', type=str, help='Output directory (has to be accessible by HPO workers).')
  parser.add_argument('-i', '--interface', type=str, default="127.0.0.1",
                      help='Communication interface (default:  %(default)s).')
  parser.add_argument('-p', '--port', type=int, default=57945,
                      help='Server port (default:  %(default)d).')
  parser.add_argument('-mib', '--min-budget', type=float, default=1,
                      help='Minimum epochs used during the optimization (default:  %(default)d).')
  parser.add_argument('-mab', '--max-budget', type=float, default=9,
                      help='Maximum epochs used during the optimization (default:  %(default)d).')
  parser.add_argument('-it', '--iterations', type=int, default=5,
                      help='Number of iterations performed by the optimizer (default:  %(default)d).')
  parser.add_argument('-w', '--workers', type=int, default=1,
                      help='Number of HPO workers (default:  %(default)d). Server waits until the specified number '
                           'of workers is connected.')

  parser.add_argument('--prev-output', type=str,
                      help='Previous output directory to be used to warmstart the HPO algorithm. '
                           'Makes sense only if exactly the same config space definition is used.')
  parser.add_argument('--run-id', type=str, default="0",
                      help='A unique run id for this optimization run (default:  %(default)s)')
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  copy_file(args.configspace, os.path.join(args.output, CONFIGSPACE_FILENAME))

  NS = hpns.NameServer(run_id=args.run_id, host=args.interface, port=args.port, working_directory=args.output)
  ns_host, ns_port = NS.start()

  get_configspace = _exec(args.configspace, "get_configspace")

  result_logger = hpres.json_result_logger(directory=args.output, overwrite=True)

  previous_run = None
  if args.prev_output:
    previous_run = hpres.logged_results_to_HBS_result(args.prev_output)

  bohb = BOHB(configspace=get_configspace(),
              run_id=args.run_id, host=args.interface, nameserver=args.interface, nameserver_port=args.port,
              min_budget=args.min_budget, max_budget=args.max_budget, result_logger=result_logger,
              previous_result=previous_run
              )
  print("Waiting for %d worker(s)." % args.workers)
  res = bohb.run(n_iterations=args.iterations, min_n_workers=args.workers)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  id2config = res.get_id2config_mapping()
  incumbent = res.get_incumbent_id()

  output_path = os.path.join(args.output, OUTPUT_FILENAME)
  with open(output_path, "w") as f:
    f.write('Best found configuration %s: %s\n' % (incumbent, id2config[incumbent]['config']))
    f.write('A total of %i unique configurations where sampled.\n' % len(id2config.keys()))
    f.write('A total of %i runs where executed.\n' % len(res.get_all_runs()))
    f.write('Total budget corresponds to %.1f full function evaluations.\n' % (
      sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
  print('Best configuration saved to %s.' % output_path)
