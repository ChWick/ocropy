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

# folds setup
parser.add_argument("--root_dir", type=str, default="",
                    help="Expects Train and Eval dir as sub dir")
parser.add_argument('--network', type=str, required=True,
                    help="Network structure that shall be used for training. Eg. 'cnn=60:3x3,pool=2x2,lstm=100")


# parse args and clean data
args = parser.parse_args()

args.root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
train_dir = os.path.join(args.root_dir, "Train")
eval_dir = os.path.join(args.root_dir, "Eval")

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
        if line is None:
            time.sleep(0.1)
        else:
            yield line


def iters_for_lines(lines):
    return int(np.interp(lines, [60, 100, 150, 250, 500, 1000], [10000, 13000, 15000, 20000, 25000, 30000]))


line_dirs_ = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and int(d)]
print("Found line dirs %s" % line_dirs_)
if len(line_dirs_) == 0:
    raise Exception("No line dirs found in Train dir '%s'" % train_dir)


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
        ]):

        print(line)

        if line.startswith("Evaluation result:"):
            evaluation_result = line

    return evaluation_result

def generate_params(line_dirs_):
    return [
            {"eval_data": eval_dir,
            "train_data": os.path.join(train_dir, line_dir_),
            "run": args.run,
            "ntrain": iters_for_lines(int(line_dir_)),
            "network": args.network,
            } for line_dir_ in line_dirs_
            ]

all_params = generate_params(line_dirs_)

pool = multiprocessing.Pool(processes=len(all_params))
evaluation_results = pool.map(run_single, all_params)
pool.close()

print("Evaluation result")
print("\n".join(evaluation_results))
