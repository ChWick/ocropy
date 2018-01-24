import os, re
import argparse
import ocrolib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--models", default=[], nargs="*", help="input models")
parser.add_argument("--files", default=[], nargs="*", help="input file")
parser.add_argument("--gt", default=[], nargs="*", help="group truth files")
parser.add_argument("--output_dir", help="output dir")

args = parser.parse_args()
args.files = ocrolib.glob_all(args.files)
args.models = ocrolib.glob_all(args.models)
args.gt = ocrolib.glob_all(args.gt)

re_params = re.compile("^.*_bs([\d]+)_.*-(\d+)\.pyrnn\.gz$")
re_acc = re.compile("^[01]\.\d+$")

def model_params(filename):
    match = re_params.match(filename)
    if match is not None:
        return match.group(1), match.group(2)

    return None


for f in args.models:
    res = model_params(f)
    if res is not None:
        batch_size, time = res
        os.system("./ocropus-rpred -T -n -Q0 -m %s %s" % (f, " ".join(args.files)))
        output = subprocess.check_output("./ocropus-errs %s" % (" ".join(args.gt)), shell=True)
        for line in output.split("\n"):
            match = re_acc.match(line.strip()) 
            if match is not None:
                with open(os.path.join(args.output_dir, "bs%s.txt" % batch_size), 'a') as ofile:
                    ofile.write("%s %s\n" % (time, float(line.strip())))


