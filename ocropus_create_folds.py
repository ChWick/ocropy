import argparse
import ocrolib
import numpy as np
import random
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--files", nargs="+", required=True,
                    help="Directory to the data")
parser.add_argument("--output", type=str, required=True,
                    help="Directory where to put the data")
parser.add_argument("--n_folds", type=int, default=5,
                    help="Number of folds to create")
parser.add_argument("--n_lines", type=int, required=True)
parser.add_argument("--remove_existing", action="store_true",
                    help="Remove the existing files if already existent, else the program will raise an exception")
parser.add_argument("--postfix_dir", type=str, default="data")

args = parser.parse_args()
args.output = os.path.join(os.path.abspath(os.path.expanduser(args.output)), str(args.n_lines), args.postfix_dir)

if args.n_lines % args.n_folds != 0:
    raise Exception("The number of lines must be dividable by the number of folds: {} %% {} = {} != 0".
                    format(args.n_lines, args.n_folds, args.n_lines / args.n_folds))

print("Creating dirs for folds")
if not os.path.exists(args.output):
    os.makedirs(args.output)

for fold in range(args.n_folds):
    fold_dir = os.path.join(args.output, str(fold))
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)
    else:
        existing_files = len(os.listdir(fold_dir)) > 0
        if args.remove_existing:
            print("Removing existing files from {}".format(fold_dir))
            shutil.rmtree(fold_dir)
            os.mkdir(fold_dir)
        else:
            raise Exception("Fold dir not empty. Use --remove_existing to clear the directory at {}".format(fold_dir))


print("Empty folds dir set up at {}".format(args.output))

print("Loading all lines")
inputs = ocrolib.glob_all(args.files)
if len(inputs) < args.n_lines:
    raise Exception("Insufficient number of lines: Available = {}, Required = {}".format(len(inputs), args.n_lines))

indices = range(len(inputs))
random.shuffle(indices)
subset = [inputs[i] for i in indices[:args.n_lines]]

print("Loaded {} lines".format(len(inputs)))
fold_size = args.n_lines // args.n_folds
print("Size of a fold is {}".format(fold_size))

for fold in range(args.n_folds):
    fold_dir = os.path.join(args.output, str(fold))
    print("Working on {}".format(fold_dir))
    for fname in subset[fold * fold_size:(fold + 1) * fold_size]:
        base, _ = ocrolib.allsplitext(fname)
        txt_name = base + ".gt.txt"

        if not os.path.exists(fname):
            raise Exception("Missing file {}".format(txt_name))
        if not os.path.exists(txt_name):
            raise Exception("Missing file {}".format(txt_name))

        img_basename = os.path.basename(fname)
        txt_basename = os.path.basename(txt_name)

        shutil.copyfile(fname, os.path.join(fold_dir, img_basename))
        shutil.copyfile(txt_name, os.path.join(fold_dir, txt_basename))

print("All folds set up")





