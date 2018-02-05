import subprocess
import multiprocessing
import argparse
import os
import numpy as np
import sys

parser = argparse.ArgumentParser()

# global setup
parser.add_argument("--run", type=str, default="",
                    help="All scripts commands will be passed to this command. {threads} will be replaced with the "
                         "number of threads that are required for a single specific task. "
                         "E. g. use 'srun --cpus-per-task {threads}' for slurm usage")
parser.add_argument("--python", type=str, default="python2",
                    help="Path to a python executable, may be in a venv!")
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--skip_eval", action="store_true")

# folds setup
parser.add_argument("--root_dirs", type=str, required=True, nargs="+",
                    help="Expects Train and Eval dir as sub dir")
parser.add_argument('--network', type=str, required=True,
                    help="Network structure that shall be used for training. Eg. 'cnn=60:3x3,pool=2x2,lstm=100")


# parse args and clean data
args = parser.parse_args()

ocropus_crossfold_args = []
if args.skip_train:
    ocropus_crossfold_args.append("--skip_train")
if args.skip_test:
    ocropus_crossfold_args.append("--skip_test")
if args.skip_eval:
    ocropus_crossfold_args.append("--skip_eval")


args.root_dirs = [os.path.abspath(os.path.expanduser(rd)) for rd in args.root_dirs]
train_dirs = [os.path.join(rd, "Train") for rd in args.root_dirs]
eval_dirs = [os.path.join(rd, "Eval") for rd in args.root_dirs]

for train_dir, eval_dir in zip(train_dirs, eval_dirs):
    if not os.path.exists(train_dir):
        raise Exception("Expected train dir at '%s'" % train_dir)

    if not os.path.exists(eval_dir):
        raise Exception("Expected eval dir at '%s'" % eval_dir)

def run_cmd(threads):
    if args.run:
        formatted = args.run.format(threads=threads)
        return formatted.split()
    else:
        return []

def run(command, process_out_list=[]):
    print(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    process_out_list.append(process)
    while True:
        line = process.stdout.readline().rstrip().decode("utf-8")

        # check if process has finished
        if process.poll() is not None:
            break

        # check if output is present
        if line is None or len(line) == 0:
            time.sleep(0.1)
        else:
            yield line


def iters_for_lines(lines):
    return int(np.interp(lines, [60, 100, 150, 250, 500, 1000], [10000, 13000, 15000, 20000, 25000, 30000]))

def line_dirs_for_train_dir(train_dir):
    line_dirs_ = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and int(d)]
    print("Found line dirs %s" % line_dirs_)
    if len(line_dirs_) == 0:
        raise Exception("No line dirs found in Train dir '%s'" % train_dir)

    return line_dirs_


def run_single(run_params):
    print("Running single with params: ")
    print(run_params)
    
    evaluation_result = ""
    for line in run(["python3", "ocropus-crossfold-best-model.py",
        "--tensorflow",
        "--root_dir", run_params["train_data"],
        "--eval_data", run_params["eval_data"],
        "--ntrain", str(run_params["ntrain"]),
        "--network", run_params["network"],
        "--run", run_params["run"],
        "--python", args.python,
        ] + ocropus_crossfold_args):

        print(line)

        if line.startswith("Evaluation result:"):
            evaluation_result = line

    return run_params, evaluation_result

def generate_params(train_dir, eval_dir):
    line_dirs_ = line_dirs_for_train_dir(train_dir)
    return [
            {"eval_data": eval_dir,
            "train_data": os.path.join(train_dir, line_dir_),
            "run": args.run,
            "ntrain": iters_for_lines(int(line_dir_)),
            "network": args.network,
            } for line_dir_ in line_dirs_
            ]

all_params =[]

for train_dir, eval_dir in zip(train_dirs, eval_dirs):
    all_params += generate_params(train_dir, eval_dir)

pool = multiprocessing.Pool(processes=len(all_params))
evaluation_results = pool.map(run_single, all_params)
pool.close()

print("Evaluation result")
for run_params, evaluation_result in evaluation_results:
    print("Result of %s:\t%s" % (run_params["train_data"], evaluation_result))
