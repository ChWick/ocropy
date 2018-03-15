import argparse
import multiprocessing
import sys
import os
import re
import shutil
import subprocess
import time
import codecs
#import ocrolib.tensorflow.model as tf_model

# only one thread to train, do not use GPU
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser()

# global setup
parser.add_argument("--dry_run", action="store_true",
                    help="Only print the run commands of crossfold computation, but do not execute them")
parser.add_argument("--run", type=str, default="",
                    help="All scripts commands will be passed to this command. {threads} will be replaced with the "
                         "number of threads that are required for a single specific task. "
                         "E. g. use 'srun --cpus-per-task {threads}' for slurm usage")
parser.add_argument("--n_parallel", type=int, default=-1,
                    help="Number of parallel models to swapn, if negative (default), all models will be spawned immediately.")
parser.add_argument("--python", type=str, default="python2",
                    help="Path to a python executable, may be in a venv!")

# folds setup
parser.add_argument("--root_dir", type=str, default="",
                    help="Expects a 'data' dir as sub dir. The data dir must contain a dir for each fold "
                         "labeled by a number")
parser.add_argument("--single_fold", type=str, default="",
                    help="If a positive number only run the specific fold, instead of all")
parser.add_argument("--verbose", action="store_true",
                    help="Verbose mode print all output of the several programs to the console. It is always written "
                    "to log files")

# training setup
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--rtrain", type=str, default="ocropus-rtrain")
parser.add_argument("--no_early_stopping", action="store_true")
parser.add_argument("--ntrain", type=int, default=200000,
                    help="# lines to train before stopping, default: %(default)s")
parser.add_argument("--tensorflow", action='store_true')
parser.add_argument("--load", type=str,
                    help="Load a pretrained model")
parser.add_argument("--load_pretrained", action="store_true",
                    help="Load weights of pretrained model, the output layer will be reshaped and retrained")
parser.add_argument('--network', type=str, required=True,
                    help="Network structure that shall be used for training. Eg. 'cnn=60:3x3,pool=2x2,lstm=100")

# testing setup
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--rpred", type=str, default="ocropus-rpred")
parser.add_argument("--econf", type=str, default="ocropus-econf")


# evaluate best models
parser.add_argument("--skip_eval_best", action="store_true")
parser.add_argument("--eval_data", type=str,
                    help="Path to the evaluation data")
parser.add_argument("--eval", type=str, default="ocropus-eval.py",
                    help="Evaluation program")


# parse args and clean data
args = parser.parse_args()

args.root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
args.rtrain = os.path.abspath(os.path.expanduser(args.rtrain))
args.rpred = os.path.abspath(os.path.expanduser(args.rpred))
args.econf = os.path.abspath(os.path.expanduser(args.econf))
data_dir = os.path.join(args.root_dir, 'data')
#model_settings = tf_model.Model.parse_model_settings(args.network)
#model_prefix = tf_model.Model.write_model_settings(model_settings)
model_prefix = args.network
if args.load:
    model_prefix = "pretrained=%s_%s" % (os.path.splitext(os.path.basename(args.load))[0], args.network)

best_models_dir = os.path.join(args.root_dir, '%s_best_models' % model_prefix)
fold_dir_ = "%s_folds" % model_prefix

tensorflow_arg = "--tensorflow" if args.tensorflow else ""

number_check_re = re.compile("[\d]+")

if not args.skip_eval_best:
    if not args.eval_data:
        raise Exception("Required argument '--eval_data' missing")

    args.eval_data = os.path.abspath(os.path.expanduser(args.eval_data))
    if not os.path.exists(args.eval_data):
        raise Exception("Evaluation dir does not exist at '%s'" % args.eval_data)


def run_cmd(threads):
    if args.run:
        formatted = args.run.format(threads=threads)
        return formatted.split()
    else:
        return []


def mkdir(path):
    if not os.path.exists(path):
        print("Creating dir %s" % path)
        os.makedirs(path)


def symlink(source, name, is_dir=True):
    if not os.path.exists(name):
        print("Creating symlink %s->%s" % (name, source))
        os.symlink(source, name, target_is_directory=is_dir)


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


def list_models(models_dir, file_ending="pyrnn.gz"):
    return sorted([f for f in os.listdir(models_dir) if f.startswith(model_prefix) and f.endswith(file_ending)])


if not os.path.isdir(args.root_dir):
    raise Exception("Root dir %s does not exist." % args.root_dir)

if not os.path.isdir(data_dir):
    raise Exception("Expected 'data' dir at root dir '%s': %s not found" % (args.root_dir, data_dir))

print("Using data dir at '%s'" % data_dir)

data_fold_dirs_ = [d for d in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, d)) and number_check_re.match(d)]

print("Found %d folds" % len(data_fold_dirs_))
print("Setting up training and test fold dirs")

fold_root_dir = os.path.join(args.root_dir, fold_dir_)
mkdir(fold_root_dir)


print("Fold root dir: %s" % fold_root_dir)
print("Best models dir: %s" % best_models_dir)


def setup_dirs():
    base_dir = os.path.abspath(os.curdir)

    for fold_dir_ in data_fold_dirs_:
        print("Setting up fold %s" % fold_dir_)
        fold_dir = os.path.join(fold_root_dir, fold_dir_)
        train_dir = os.path.join(fold_dir, 'train')
        test_dir = os.path.join(fold_dir, 'test')
        models_dir = os.path.join(fold_dir, 'models')

        train_fold_dirs_ = data_fold_dirs_[:]
        train_fold_dirs_.remove(fold_dir_)

        mkdir(fold_dir)
        mkdir(train_dir)
        mkdir(test_dir)
        mkdir(models_dir)

        # create relative syslinks
        os.chdir(train_dir)
        for p in train_fold_dirs_:
            symlink(os.path.join("..", "..", "..", "data", p), p)

        os.chdir(test_dir)
        symlink(os.path.join("..", "..", "..", "data", fold_dir_), fold_dir_)

    os.chdir(base_dir)


setup_dirs()

print("All directories created")

if len(args.single_fold) > 0:
    if args.single_fold not in data_fold_dirs_:
        raise Exception("Single fold dir '%s' not a valid fold: Valid folds are %s"
                        % (args.single_fold, data_fold_dirs_))

    data_fold_dirs_ = [args.single_fold]

fold_dirs = [os.path.join(fold_root_dir, fold_dir_) for fold_dir_ in data_fold_dirs_]



# training function that can be parallel
def train_single_fold(fold_dir):
    print("Running training for fold '%s'" % fold_dir)
    assert(os.path.exists(fold_dir))

    train_dir = os.path.join(fold_dir, 'train')
    test_dir = os.path.join(fold_dir, 'test')
    models_dir = os.path.join(fold_dir, 'models')
    train_log = os.path.join(fold_dir, '%s_train.log' % model_prefix)

    # clear models
    for old_model in list_models(models_dir, file_ending=""):
        os.remove(os.path.join(models_dir, old_model))

    with open(train_log, 'w') as train_log_file:
        process = []
        if args.no_early_stopping:
            early_stopping_args = []
        else:
            early_stopping_args = [
                "--validation", os.path.join(test_dir, "*", "*.png")
            ]
        for line in run(run_cmd(1) +
                        [args.python, args.rtrain, tensorflow_arg,
                         "--preload",
                         "--network", args.network,
                         "--threads", "0",
                         ] + early_stopping_args + [
                         # "--batch_size", 1,
                         "--ntrain", str(args.ntrain),
                         "--codec", os.path.join(train_dir, "*", "*.gt.txt")] +
                        (["--load", args.load] if args.load else []) +
                        (["--load_pretrained"] if args.load_pretrained else []) +
                        [
                            "-o", os.path.join(models_dir, model_prefix),
                            os.path.join(train_dir, "*", "*.png"),
                        ], process):
            if args.verbose:
                print("Fold %s: %s" % (fold_dir, line.strip()))
            train_log_file.write(line + "\n")
            train_log_file.flush()

        process = process[0]
        process.communicate()
        if process.returncode != 0:
            print("Error in training step. exitting.")
            raise Exception("Error in training step")


    print("Finished training for fold '%s'" % fold_dir)


print("Starting the training")

if args.skip_train:
    print("Skipping")
elif args.n_parallel == 1:
    list(map(train_single_fold, fold_dirs))
else:
    multi_pool = multiprocessing.Pool(processes=args.n_parallel if args.n_parallel > 0 else len(fold_dirs))
    multi_pool.map(train_single_fold, fold_dirs)
    multi_pool.close()

print("Training Finished")


print("Running models on test set")


if not args.skip_test:
    # prerequisites, all splits have equal amount of models
    number_of_models = len(list_models(os.path.join(fold_dirs[0], 'models')))
    for fold_dir in fold_dirs:
        this_number_of_models = len(list_models(os.path.join(fold_dirs[0], 'models')))
        if this_number_of_models != number_of_models:
            raise Exception("Mismatch in number of models of fold '%s': %d  vs %d"
                            % (fold_dir, this_number_of_models, number_of_models))

    if number_of_models == 0:
        raise Exception("No models to test found in '%s' with prefix '%s'!" % (best_models_dir, model_prefix))

    mkdir(best_models_dir)


def copy_model(model, models_dir, output_name, output_dir):
    assert(os.path.exists(models_dir))
    assert(os.path.exists(output_dir))
    model_files = [f for f in os.listdir(models_dir) if f.startswith(model)]
    if len(model_files) == 0:
        raise Exception("No model files with prefix {} found in {}".format(model, models_dir))

    for model_file in model_files:
        output_file = model_file.replace(model, output_name)
        src = os.path.join(models_dir, model_file)
        target = os.path.join(output_dir, output_file)
        print("Copy model from %s to %s" % (src, target))
        shutil.copyfile(src, target)


eval_lines_to_extract = ["errors", "missing", "total", "err", "errnomiss"]


def extract_line_data(output):
    line_data = {}
    for line in output.split("\n"):
        split = line.split()
        if len(split) <= 1:
            continue

        if split[0] in eval_lines_to_extract:
            line_data[split[0]] = split[1]

    return line_data


def test_single_fold(fold_dir):
    fold = os.path.basename(os.path.normpath(fold_dir))
    print("Running testing for fold '%s'" % fold_dir)
    assert(os.path.exists(fold_dir))

    test_dir = os.path.join(fold_dir, 'test')
    models_dir = os.path.join(fold_dir, 'models')
    test_log = os.path.join(fold_dir, '%s_test.log' % model_prefix)
    eval_csv = os.path.join(fold_dir, '%s_eval.csv' % model_prefix)

    with codecs.open(test_log, 'w', "utf-8") as test_log_file, open(eval_csv, 'w') as eval_csv_file:
        all_models = list_models(models_dir)
        eval_csv_file.write("model," + ",".join(eval_lines_to_extract) + "\n")

        current_best_model = ""
        current_best_error = 1000

        best_model_re = re.compile("^best_model_path \(score\): (.*) \(([0-9.]+)\)$")

        all_model_paths = [os.path.join(models_dir, model) for model in all_models]
        for line in run(run_cmd(16) + [args.python, args.eval,
                                    "--models"] + all_model_paths+
                                   ["--ground_truth", os.path.join(test_dir, "*", "*.gt.txt"),
                                    "--files", os.path.join(test_dir, "*", "*.png"),
                                    "--batch_size", "50",
                                    "--threads", str(16),
                                    "--output_best_only",
                                    ]):

            if args.verbose:
                print("Fold %s: %s" % (fold_dir, line.strip()))

            match = best_model_re.match(line.strip())
            if match:
                current_best_model = all_models[all_model_paths.index(match.group(1))]
                current_best_error = float(match.group(2))

        print("Fold %s: Best model is %s with error of %s %%" % (fold_dir, current_best_model, current_best_error))
        copy_model(current_best_model, models_dir, "%s_%s.pyrnn.gz" % (model_prefix, fold), best_models_dir)
        print("Copied best model to %s" % best_models_dir)

    print("Finished testing for fold '%s'" % fold_dir)
    return current_best_model, current_best_error


if args.skip_test:
    print("Skipping")
    best_model_accuracy = None
elif args.no_early_stopping:
    if args.n_parallel == 1:
        best_model_accuracy = list(map(test_single_fold, fold_dirs))
    else:
        multi_pool = multiprocessing.Pool(processes=args.n_parallel if args.n_parallel > 0 else len(fold_dirs))
        best_model_accuracy = multi_pool.map(test_single_fold, fold_dirs)
        multi_pool.close()

    if best_model_accuracy:
        print("Best model\tAcc")
        for best_model, best_error in best_model_accuracy:
            print("%s\t%f" % (best_model, best_error))
else:
    print("Copying best models of early stopping")
    for fold_dir in fold_dirs:
        fold = os.path.basename(os.path.normpath(fold_dir))
        models_dir = os.path.join(fold_dir, 'models')
        copy_model("{}best_model".format(model_prefix), models_dir, "%s_%s" % (model_prefix, fold), best_models_dir)


print("Testing Finished")


print("Starting evaluation of best models")

best_models = list_models(best_models_dir)

if len(best_models) == 0:
    raise Exception("No best models found in '%s'!" % best_models_dir)

elif args.skip_eval_best:
    print("Skipping")
else:
    with codecs.open(os.path.join(args.root_dir, "%s_eval.csv" % model_prefix), 'w', 'utf-8') as eval_file:

        # all models must be evaluated at the same time to enable voting
        threads_per_model = 4
        total_threads = len(best_models) * threads_per_model
        evaluation_result = None
        for line in run(run_cmd(total_threads) + [args.python, args.eval,
                                    "--models"] + [os.path.join(best_models_dir, m) for m in best_models] +
                                   ["--ground_truth", os.path.join(args.eval_data, "*.gt.txt"),
                                    "--files", os.path.join(args.eval_data, "*.png"),
                                    "--batch_size", "50",
                                    "--threads", str(total_threads),
                                    "--output_evaluation",
                                    ]):

            if line.startswith("Evaluation result:"):
                evaluation_result = line

        if evaluation_result is None:
            raise Exception("Evaluation did not yield an parsable output")

        eval_file.write(evaluation_result)
        print(evaluation_result)

print("Evaluation finished")
