#!/usr/bin/env python
import argparse
import logging
import csv
import os
from itertools import count
import glob
from typing import Dict

import pandas as pd
import numpy as np

from netstat import Stat


class ThreshRes:
    def __init__(self, target: int, data: pd.DataFrame):
        self.target = target
        self.data: pd.DataFrame = data.set_index("scale")
        return

    def get_percep(self, scale: float) -> (float, float):
        percep = self.data[["perceptability", "clp_perceptability"]]
        scale_name: str = f"s{scale}"
        if scale_name == "s0.3":
            scale_name = "s0.30000000000000004"

        ret = " c".join(f"p={v}" for v in percep.loc[scale_name])
        return ret

    def __call__(self, scale: str) -> (float, float):
        return self.get_percep(float(scale.split("-")[0]) / 100)


class TrainResult:
    def __init__(self, name: str):
        self.name = name
        self.epochs = None
        self.topk = {f"top_{k}": list() for k in (1, 5)}
        self.success_rate = list()
        self.poisoned_samples = list()
        self.p2targ_count = list()
        self.mis_pred_rate = list()

    def finalStats(self):
        return (
            self.name,
            self.success_rate[-1],
            self.epochs,
            np.mean(self.poisoned_samples),
        )

    def getStats(self):
        max_sr = max(self.success_rate)
        max_epc = self.success_rate.index(max_sr)
        ps = sum(self.poisoned_samples[: max_epc + 1])
        return self.name, max_sr, max_epc, ps

    def __str__(self):
        if self.epochs is None:
            return f"{self.name}: Empty"
        return f"{self.name}: " + " - ".join(
            [str(success_rate) for success_rate in self.success_rate]
        )

    def __call__(self, *args, **kwargs):
        append_val = ""
        if self.epochs < 1:
            logging.warning(
                "Only showing info for un-trained victim (epoch " "0) in %s", self.name
            )
            append_val += " <- Epoch 0"

        max_sr = max(self.success_rate)
        max_epc = self.success_rate.index(max_sr)
        out = (
            f"{self.name}: {max_sr}% success rate at epoch {max_epc} "
            f"({sum(self.poisoned_samples[:max_epc + 1])})"
        )
        if max_epc != self.epochs:
            out += (
                f"; {self.success_rate[-1]}% success rate at end" f" (e{self.epochs})"
            )
        out += (
            f" with {sum(self.poisoned_samples)} trigs"
            f" {np.mean(self.poisoned_samples)}"
            f"({np.std(self.poisoned_samples)})"
        )

        if len(self.topk["top_1"]) == 1:
            return "No adversarial training"

        if min(self.topk["top_1"][1:]) < self.topk["top_1"][0] - (
            self.topk["top_1"][0] * 0.5
        ):
            min_epc = self.topk["top_1"].index(min(self.topk["top_1"][1:]))
            out += f'\n\tt1 < {self.topk["top_1"][0]} at epoch {min_epc}'

        for x in args:
            append_val += f" {x}"

        return out + append_val

    def __gt__(self, other):
        return self.compare(other, lambda x, y: x > y)

    def __lt__(self, other):
        return self.compare(other, lambda x, y: x < y)

    def __ge__(self, other):
        return self.compare(other, lambda x, y: x >= y)

    def __le__(self, other):
        return self.compare(other, lambda x, y: x <= y)

    def __eq__(self, other):
        return self.compare(other, lambda x, y: x == y)

    def compare(self, other, comparison):
        # allow comparison with explicit success rate
        if type(other) == float or type(other) == int:
            return comparison(self.success_rate[-1], other)

        if self.epochs != other.epochs:
            logging.warning(
                "Comparing runs with different epoch values %s " "and %s",
                self.name,
                other.name,
            )
            if self.epochs == 0 or other.epochs == 0:
                logging.warning(
                    "Comparing with an untrained victim run %s! "
                    "Choosing whichever has highest in the FINAL "
                    "EPOCH",
                    self.name if self.epochs == 0 else other.name,
                )
                return comparison(self.success_rate[-1], other.success_rate[-1])

            epc = min(self.epochs, other.epochs)
        else:
            epc = self.epochs
        logging.debug(
            "Comparing %f and %f", self.success_rate[epc], other.success_rate[epc]
        )
        return comparison(self.success_rate[epc], other.success_rate[epc])

    def get_max(self):
        return max(self.success_rate)

    def compare_all(self, other, comparison):
        res = list()
        assert type(self) == type(other), "Can only compare with runs"
        for epoch in range(min(self.epochs, other.epochs)):
            res.append(
                self.name
                if comparison(self.success_rate[epoch], other.success_rate[epoch])
                else other.name
            )

        return res, " - ".join(res)

    def add_result(self, entry, stat):
        if entry[:3] == "top":
            self.topk[entry].append(float(stat))
        elif entry == "epoch":
            self.epochs = 0 if self.epochs is None else self.epochs + 1
        elif entry == "success_rate":
            self.success_rate.append(float(stat))
        elif entry == "poisoned_samples":
            self.poisoned_samples.append(int(stat))
        elif entry == "p2targ_count":
            self.p2targ_count.append(int(stat))
        elif entry == "mis_pred_rate":
            self.mis_pred_rate.append(float(stat))
        else:
            logging.warning("Found untracked stat %s", entry)
        return


# type alias
SortedData = Dict[str, TrainResult]
PercepData = Dict[int, ThreshRes]


def load_run(file_name: str) -> SortedData:
    """

    :param file_name: shortened results file to load
    :return: dict of runs
    """
    with open(file_name, "r") as res_file:
        reader = csv.DictReader(res_file)
        runs = {
            field.split("_")[0]: TrainResult(field.split("_")[0])
            for field in reader.fieldnames
        }
        for epoch in reader:
            for entry in epoch:
                logging.debug("entry is %s", entry)
                if epoch[entry] == "":  # didn't get to this epoch, skip
                    continue
                runs[entry.split("_")[0]].add_result(
                    "_".join(entry.split("_")[1:]), epoch[entry]
                )
    return runs


def load_percep(percep_path: str) -> PercepData:
    if percep_path is None:
        return dict()

    found_files = glob.glob(f"{percep_path}/*thresh")

    thresh_res: PercepData = dict()

    def get_target(fname: str) -> int:
        sname = fname.split("_")
        for c in sname:
            if c[0] == "t":
                return int(c[1:])
        print(f"Failed to find target in {fname}")
        raise IndexError

    for thresh_f in found_files:
        targ = get_target(thresh_f)
        thresh_res[targ] = ThreshRes(targ, pd.read_csv(thresh_f))

    return thresh_res


def load_sort(file_name: str) -> SortedData:
    """Load a shortened results file, then sort by success rates.
    :param file_name:  file to load
    :return: sorted dict of runs; from highest success rate to lowest
    """
    results = load_run(file_name)
    return {
        k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)
    }


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error(f"File {arg} does not exist")
    else:
        return str(arg)


def setup_args(parser):
    parser.add_argument(
        "--max_display",
        "-d",
        default=6,
        type=int,
        help="How many sorted runs to display",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Silence warnings")
    parser.add_argument(
        "--max_scale",
        "-s",
        default=None,
        type=float,
        help="Maximum scale to display for train/test",
    )
    parser.add_argument(
        "--percep",
        "-p",
        default=None,
        type=str,
        help="Logs to use for perceptability info",
    )


def below_max(name: str, max_scale: float) -> bool:
    if max_scale is None:
        return True

    train, test = name.split("-")

    if float(train) > max_scale * 100:
        return False
    if float(test) > max_scale * 100:
        return False

    return True


def process_file(result_file, args):
    print(f"{result_file}:")
    perceps: PercepData = load_percep(args.percep)
    sorted_res: SortedData = load_sort(result_file)

    target: int = int(result_file.split("/")[-1].split("-")[0])

    if perceps.get(target) is not None:
        print(
            "\n".join(
                sorted_res[run](perceps[target](run))
                for idx, run in enumerate(sorted_res)
                if idx < args.max_display and below_max(run, args.max_scale)
            )
        )
    else:
        print(
            "\n".join(
                sorted_res[run]()
                for idx, run in enumerate(sorted_res)
                if idx < args.max_display and below_max(run, args.max_scale)
            )
        )

    keys = [k for k in sorted_res.keys() if below_max(k, args.max_scale)]
    most_successful = keys[0]
    if sorted_res[most_successful] < 70.0 and not args.quiet:
        print(
            "LOW SUCCESS RATES, CONSIDER NEW VERSION OR LARGER SCALES: "
            "Did the generator converge to a good state?"
        )
    return {k: sorted_res[k] for k in keys}


def accumulate_stats(
    result_file,
    sorted_res,
    overall_stats: dict,
    max_cfg: list,
    all_time_max: list,
    all_time_epc: list,
    all_time_trig: list,
    quiet: bool,
):
    max_overall = 0
    max_run = None

    for run in sorted_res:
        stats = sorted_res[run].finalStats()
        r_name = stats[0]

        if stats[2] == 0:  # likely incomplete, skip
            if not quiet:
                print("Skipping incomplete", r_name)
            continue

        if stats[1] > max_overall:
            max_run = run
            max_overall = stats[1]

        if overall_stats.get(r_name):
            overall_stats[r_name]["sr"] += stats[1]
            overall_stats[r_name]["ps"].accumulate(stats[3], stats[2] + 1)
            overall_stats[r_name]["ep"].append(stats[2])

        else:
            overall_stats[r_name] = {
                "sr": Stat(0, average=True, var=True),
                "ps": Stat(0, average=True, var=True),
                "ep": [stats[2]],
            }
            # do it this way for the internal stat counter
            overall_stats[r_name]["sr"] += stats[1]
            overall_stats[r_name]["ps"].accumulate(stats[3], stats[2] + 1)

    if max_run is not None:
        stats = sorted_res[max_run].finalStats()
        if stats[2] != 0:
            max_cfg.append(
                f"  Target "
                f'{result_file.split("/")[-1].split("-")[0]}: '
                f"{sorted_res[max_run]()}"
            )

            all_time_max.append(stats[1])
            all_time_epc.append(stats[2])
            all_time_trig.append(stats[3])
    else:
        if not quiet:
            print("empty??")


def print_stats(
    overall_stats: dict,
    max_cfg: list,
    all_time_max: list,
    all_time_epc: list,
    all_time_trig: list,
    max_display: int = None,
):
    print(f"For each of {len(max_cfg)} targets:")
    print("\n".join(max_cfg))

    print(
        f"Average Best: {np.mean(all_time_max)}% "
        f"({np.std(all_time_max):.2f}) "
        f"at {np.mean(all_time_epc)} epochs "
        f"with {np.mean(all_time_trig)} ({np.std(all_time_trig)}) "
        f"+{max(all_time_trig)} -{min(all_time_trig)} triggers"
    )
    # print('\n')

    displayed = count()

    for stat in overall_stats:
        if max_display is not None and next(displayed) == max_display:
            break

        print(f"Over {overall_stats[stat]['sr'].count} classes: ")
        sr, sr_v = overall_stats[stat]["sr"].get_detailed()
        ps, ps_v = overall_stats[stat]["ps"].get_detailed()

        # print(max(overall_stats[stat]['sr'].dat))
        # print(min(overall_stats[stat]['sr'].dat))
        # print(overall_stats[stat]['sr'].dat)
        # print(overall_stats[stat]['ps'].dat)
        # print(overall_stats[stat]['ep'])

        print(
            f"  {stat}: {sr:.2f}% ({sr_v:.3f}) "
            f"-- {ps:.1f} ({ps_v:.1f}) "
            f" @ {min(overall_stats[stat]['ep'])}/"
            f"{max(overall_stats[stat]['ep'])}/"
            f"{np.mean(overall_stats[stat]['ep'])}\n"
        )


if __name__ == "__main__":
    FORMAT = "%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]"
    logging.basicConfig(
        level=logging.ERROR, format=FORMAT, handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=lambda x: is_valid_file(parser, x),
        help="Shortened result file to process",
    )
    setup_args(parser)
    args = parser.parse_args()

    # output
    overall_stats = dict()
    max_cfg = list()
    all_time_max = list()
    all_time_epc = list()
    all_time_trig = list()

    for result_file in args.result_files:
        sorted_res = process_file(result_file, args)

        # max_overall = max([sorted_res[k].get_max() for k in sorted_res])

        accumulate_stats(
            result_file,
            sorted_res,
            overall_stats,
            max_cfg,
            all_time_max,
            all_time_epc,
            all_time_trig,
            args.quiet,
        )

        # print('\n\n')

    print_stats(
        overall_stats,
        max_cfg,
        all_time_max,
        all_time_epc,
        all_time_trig,
        args.max_display,
    )
