"""Process output files to shorten the number of classes displayed"""

import logging
import argparse
import csv
import os


def gen_keys(row, prefix):
    """Generate the base keys for this row"""
    # first, figure out which keys are label independent (eg, epoch vs margin)
    short_row = dict()
    base_keys = []
    for key in row.keys():
        if key.split("_")[0] == "top":  # special case (top_k) skip
            short_row[f"{prefix}_{key}"] = row[key]
            continue

        try:
            # key pattern is [base_key]_[label] ; since label is a number,
            # this tells us whether this key repeats for every label
            int(key.split("_")[-1])
        except ValueError:
            # the key is label independent, no processing needed: just copy
            # the value over
            short_row[f"{prefix}_{key}"] = row[key]
            continue

        base_key = "_".join(key.split("_")[:-1])
        if base_key not in base_keys:
            base_keys.append(base_key)

    return base_keys, short_row


def process_row(row, prefix, opts):
    """Extract meaningful data from the current row"""
    base_keys, short_row = gen_keys(row, prefix)

    # for non-p2targ elements, include [label_count] elements, then stop.
    #
    # for p2targ elements, include [label_count] elements, but count all
    # elements that are above the cutoff before stopping
    for base_key in base_keys:
        # extract all elements matching base_key pattern, and sort
        elems = {k: v for k, v in row.items() if k.startswith(base_key)}
        # for the elements below, we want the SMALLEST values
        reverse = base_key.split("_")[0] not in ["avg", "std", "max", "min"]

        cutoff_count = 0
        for i, (k, v) in enumerate(
            sorted(elems.items(), key=lambda item: float(item[1]), reverse=reverse)
        ):

            if float(v) >= opts.cutoff:
                cutoff_count += 1

            if i < opts.label_count:
                short_row[f"{prefix}_{base_key}_l{i}"] = k.split("_")[-1]
                short_row[f"{prefix}_{base_key}_{i}"] = v
            elif not reverse:  # base_key in avg, std, max, min
                break
            elif opts.cutoff > float(v):  # p2targ
                break

        if reverse:
            short_row[f"{prefix}_{base_key}_count"] = cutoff_count
            print(
                f'(E{float(short_row[f"{prefix}_epoch"]):.0f}) '
                f"{prefix}_{base_key}:"
                f' {short_row[f"{prefix}_{base_key}_count"]}'
                f' -- {short_row[f"{prefix}_success_rate"]}%'
            )

    return short_row


def extract_info(fname):
    """From a file name, extract the train/test scales"""
    # strip path and split
    # filename: train_[dataset]_t[target]_[bias]_te[test_scale]_*
    fname = fname.replace("cifar10_pgd", "cifar10-pgd")
    fname = fname.replace("cifar10_r", "cifar10-r")
    split_fname = fname.split("/")[-1].split("_")
    logging.debug("-----Processing %s ", " ".join(split_fname))

    if split_fname[0] != "train":
        logging.debug("Is black box or base")
        split_fname.pop(0)
    split_fname.pop(0)  # [dataset]

    dataset = split_fname[0]
    # if split_fname[1] == 'pgd':
    #     dataset = '_'.join([dataset, 'pgd'])
    #     split_fname.pop(0)  # t[target]
    logging.debug("Dataset is %s", dataset)
    split_fname.pop(0)  # t[target]

    target_label = split_fname[0][1:]
    logging.debug("target is %s", target_label)
    split_fname.pop(0)  # [bias]

    bias = split_fname[0]
    logging.debug("bias is %s", bias)
    split_fname.pop(0)  # te[test_scale]

    test_scale = split_fname[0][2:]  # format is: 'te[float]'
    logging.debug("test scale is %s", test_scale)
    split_fname.pop(0)  # te[test_scale]

    train_scale = split_fname[0][2:]  # format is: 'tr[float]'
    logging.debug("train scale is %s", train_scale)
    split_fname.pop(0)

    if len(split_fname[0]) >= 4 and split_fname[0][:3] == "pre":
        logging.debug("pre is %s", split_fname[0][3:])
        split_fname.pop(0)

    if len(split_fname[0]) >= 5 and split_fname[0][:4] == "post":
        logging.debug("post is %s", split_fname[0][4:])
        split_fname.pop(0)

    # split_fname.pop(0)  # [run_id]-[bias]-[train_scale]-[query_limit].csv
    logging.debug("runid: %s", split_fname[0])
    split_info = split_fname[0].split("-")
    run_id = split_info[0]
    logging.debug("type: %s", run_id)
    # run_id was: [type]-[run_id]
    if run_id == "bbox" or run_id == "base" or run_id == "noise":
        split_info.pop(0)
        run_id += split_info[0]

    if len(split_info) > 1:
        split_info.pop(0)

        checkpoint_epoch = split_info[-1].split(".")[0]  # trim csv extension
        logging.debug("Checkpoint epoch: %s", checkpoint_epoch)
    else:
        checkpoint_epoch = None
        logging.warning("Checkpoint epoch not found")

    return {
        "test_scale": test_scale,
        "dataset": dataset,
        "target": target_label,
        "bias": bias,
        "train_scale": train_scale,
        "run_id": run_id,
        "checkpoint_epoch": checkpoint_epoch,
    }


def format_name(fname):
    """Format the filename using extract info"""
    info = extract_info(fname)
    return f'{info["train_scale"]}-{info["test_scale"]}'


def extract_values(fname, opts):
    """Load and extract the values of interest"""
    prefix = format_name(fname)

    processed_dat = list()
    with open(fname, "r") as log_file:
        reader = csv.DictReader(log_file)
        for row in reader:
            processed_dat.append(process_row(row, prefix, opts))
    return processed_dat


def setup_args(parser):
    stat_group = parser.add_argument_group("Cutoff options")
    stat_group.add_argument(
        "--label_count", "-k", type=int, default=0, help="Number of labels to show"
    )
    stat_group.add_argument(
        "--cutoff", "-c", type=float, default=90, help="Success Rate cutoff"
    )


def accumulate_file(file, args, all_dat: dict, out_dirs: dict, errors: list):
    try:
        run_info = extract_info(file)
        outdir = "/".join(file.split("/")[:-1])
        logging.debug("outdir is %s", outdir)
    except IndexError:
        if not os.path.isdir(file):
            err_str = f"{file}"
        else:
            err_str = f"{file} (cannot process directories)"
        logging.error("Potential error in %s", err_str)
        errors.append(err_str)
        return False
    processed = extract_values(file, args)

    run_name = (
        f'{run_info["target"]}'
        f'-{run_info["bias"]}'
        f'-{run_info["run_id"]}'
        f'-{run_info["checkpoint_epoch"]}'
    )
    logging.info("Processing data for %s", run_name)

    if run_name not in all_dat:
        all_dat[run_name] = list()
        out_dirs[run_name] = outdir

    # ensure all_dat list is long enough
    while len(processed) > len(all_dat[run_name]):
        all_dat[run_name].append(dict())

    # merge new data into list
    for epoch, dat in enumerate(processed):
        all_dat[run_name][epoch].update(dat)
    logging.debug("Finished %s", file)
    return True


def save_shortened(all_dat: dict, out_dirs: dict, args):
    for run_name in all_dat.keys():
        outdir = args.out or out_dirs[run_name]
        outdir += "/shortened"
        os.makedirs(outdir, exist_ok=True)
        out_fname = f"{outdir}/{run_name}.csv"
        print(f"Saving {out_fname}")

        with open(out_fname, "w+") as out_file:
            writer = csv.DictWriter(out_file, all_dat[run_name][0].keys())
            writer.writeheader()
            for epoch in all_dat[run_name]:
                writer.writerow(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Files to shorten and combine", nargs="+")
    parser.add_argument("--out", type=str, help="Output directory", default=None)

    setup_args(parser)
    args = parser.parse_args()

    FORMAT = "%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]"

    logging.basicConfig(
        level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler()]
    )

    max_epoch = 0
    all_dat = dict()  # checkpoint: [data]
    out_dirs = dict()  # checkpoint: [data]
    errors = []
    # extract and combine into a single file, for easier plotting
    for file in args.files:
        if not accumulate_file(file, args, all_dat, out_dirs, errors):
            print(f"Skipped {file}")

    save_shortened(all_dat, out_dirs, args)

    if len(errors) != 0:
        print("\n\nPotential problems with:")
        print("\n".join(errors))
