"""Sweeps various thresholds on a generator, counting the number of images
which it makes selectable at each threshold."""
import os
import csv

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import SampleImage, setup_dataset, setup_args, get_trig
from netstat import Stat


def setup():
    args = setup_args(mode="threshold")

    # train victim true to load the generator ckpt
    victim, generator, opts = setup_dataset(args, "threshold")
    device = opts["device"]
    print(device)
    normalize = opts["normalize"]
    target_loader = (
        opts["target_loader_test"]
        if not args.train_set
        else opts["target_loader_train"]
    )
    gan_noise = opts["noise_gen"]
    if opts["cutoff"] is not None:  # if none, a baseline run
        upper_bound = opts["cutoff"] + opts["cutoff_range"]
    else:
        upper_bound = None
    print(f"Upper bound is {upper_bound}")

    pgd = args.dataset[-4:] == "_pgd"

    generator.eval()
    victim.eval()

    if not args.clean:
        if args.baseline:
            run_id = "base"
            test_scales_base = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
        else:
            run_id = args.generator.split("/")[-1].split(".")[0]
            test_scales_base = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        test_scales = []
        for ts in test_scales_base:
            for i in range(1, 5):
                if ts * i not in test_scales:
                    test_scales.append(ts * i)

        test_scales.sort()
        print(f"Test Scales: {test_scales}")

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"Thresholds: {thresholds}")

    outfile = f"{args.output_directory}/{run_id}.thresh"
    sample_outdir = f"{args.output_directory}/{run_id}_tsweep"
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(sample_outdir, exist_ok=True)

    sampler = SampleImage(
        device,
        args.dataset,
        normalize,
        sample_outdir,
        opts["target_label"],
        opts["noise_gen"],
        opts["cutoff"] + opts["cutoff_range"],
        opts["norm_type"],
    )
    sampler.gen_imgs(
        f"{run_id}_tsweep", target_loader, generator, args.clip, test_scales
    )
    sampler.gen_noise(f"{run_id}_tsweep", generator, args.clip, test_scales)

    return {
        "device": device,
        "victim": victim,
        "generator": generator,
        "target_loader": target_loader,
        "normalize": normalize,
        "gan_noise": gan_noise,
        "upper_bound": upper_bound,
        "test_scales_base": test_scales_base,
        "test_scales": test_scales,
        "thresholds": thresholds,
        "outfile": outfile,
        "clip": args.clip,
        "norm_type": opts["norm_type"],
        "pgd": pgd,
    }


def perceptiveness(img, triggers):
    t_mags = torch.norm(triggers.view(triggers.size(0), -1), dim=1)
    i_mags = torch.norm(img.view(img.size(0), -1), dim=1)

    ratios = t_mags.detach().cpu() / i_mags

    return ratios.numpy()


# def get_triggers(generator, gan_noise, device, upper_bound):
#     seed = gan_noise().to(device)
#     lcount = 0  # large count
#
#     gen_imgs = generator(seed)
#
#     if upper_bound is not None:  # skip for noise-trigger baseline
#         mags = torch.norm(gen_imgs.view(gen_imgs.size(0), -1), dim=1)
#         if gen_imgs.size(0) > 1:
#             # if anything exceeded the magnitude threshold (measurable by
#             # the adversary offline) replace it with a trigger which didn't
#             # exceed the threshold
#             if (mags > upper_bound).sum().item() > 0:
#                 lcount += (mags > upper_bound).sum().item()
#                 gen_imgs[mags > upper_bound] = gen_imgs[mags <
#                                                         upper_bound][0]
#         else:  # batch size == 1
#             # in this case, we need to regenerate the trigger if it exceeds
#             # the adversary's cutoff
#             while (mags > upper_bound).sum().item() > 0:
#                 seed = gan_noise().to(device)
#                 gen_imgs = generator(seed)
#
#     return gen_imgs, lcount


def compute_margin(victim, nimg, pgd):
    if not pgd:
        confidences = victim(nimg)
    else:
        confidences, _ = victim(nimg)
    top2 = torch.topk(F.softmax(confidences, dim=1), 2)
    top2_sp = torch.split(top2[0], 1, dim=1)
    margin = top2_sp[0] - top2_sp[1]
    margin = margin.squeeze()

    return margin


class bcolors:
    RED = "\u001b[31m"
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\u001b[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def evaluate_perf(**kwargs):
    thresholds = kwargs["thresholds"]

    counts = dict()
    clp_counts = dict()
    percepts = dict()
    clp_percepts = dict()

    large_count = Stat(0, fmt=lambda x: f"; {x} too large" if x > 0 else f"")

    for c_scale in tqdm(
        kwargs["test_scales"],
        unit="scales",
        position=0,
        dynamic_ncols=True,
        desc="Evaluating",
    ):
        c_scale_key = f"s{c_scale}"
        if c_scale_key in counts:  # already done; skip
            continue
        counts[c_scale_key] = {f"{t}": 0 for t in thresholds}
        clp_counts[c_scale_key] = {f"{t}": 0 for t in thresholds}
        percepts[c_scale_key] = Stat(0, average=True)
        clp_percepts[c_scale_key] = Stat(0, average=True)

        for img, lbl in tqdm(
            kwargs["target_loader"],
            unit="Batches",
            disable=len(kwargs["target_loader"]) == 1,
            position=1,
            dynamic_ncols=True,
            desc=f"Scale {c_scale:.3f}",
        ):

            gen_imgs, lcount = get_trig(
                kwargs["generator"],
                img.size(0),
                kwargs["gan_noise"],
                kwargs["norm_type"],
                kwargs["upper_bound"],
            )
            # gen_imgs, lcount = get_triggers(kwargs['generator'],
            #                                 kwargs['gan_noise'],
            #                                 kwargs['device'],
            #                                 kwargs['upper_bound'])
            large_count += lcount - 1

            gen_imgs *= c_scale

            max_clip = torch.max(gen_imgs).detach() * kwargs["clip"]
            min_clip = torch.min(gen_imgs).detach() * kwargs["clip"]

            clp_imgs = gen_imgs.detach().clone()
            clp_imgs[clp_imgs < min_clip] = min_clip
            clp_imgs[clp_imgs > max_clip] = max_clip

            n2i_ratio = perceptiveness(img, gen_imgs)
            percepts[c_scale_key].accumulate(n2i_ratio.sum(), len(n2i_ratio))
            clp_n2i_ratio = perceptiveness(img, clp_imgs)
            clp_percepts[c_scale_key].accumulate(
                clp_n2i_ratio.sum(), len(clp_n2i_ratio)
            )

            nimg = img.to(kwargs["device"]) + gen_imgs
            cimg = img.to(kwargs["device"]) + clp_imgs
            if not kwargs["pgd"]:
                nimg = torch.stack(list(map(kwargs["normalize"], nimg)))
                cimg = torch.stack(list(map(kwargs["normalize"], cimg)))

            margin = compute_margin(kwargs["victim"], nimg, kwargs["pgd"])
            c_margin = compute_margin(kwargs["victim"], cimg, kwargs["pgd"])

            for thresh in thresholds:
                counts[c_scale_key][f"{thresh}"] += (margin < thresh).sum().item()
                clp_counts[c_scale_key][f"{thresh}"] += (c_margin < thresh).sum().item()

        ratio = counts[c_scale_key][str(thresh)] / len(kwargs["target_loader"])
        disp_vals = [f"{ratio * 100:.2f}%" for thresh in thresholds]
        disp_vals[thresholds.index(0.3)] = (
            f"{bcolors.RED}" f"{disp_vals[thresholds.index(0.3)]}" f"{bcolors.ENDC}"
        )

        ratio = clp_counts[c_scale_key][str(thresh)] / len(kwargs["target_loader"])
        clp_disp_vals = [f"{ratio * 100:.2f}%" for thresh in thresholds]
        clp_disp_vals[thresholds.index(0.3)] = (
            f"{bcolors.RED}" f"{clp_disp_vals[thresholds.index(0.3)]}" f"{bcolors.ENDC}"
        )

        tqdm.write(
            f"{c_scale_key}: "
            f'{" ".join(disp_vals)}'
            f"{str(large_count)} -- "
            f"{bcolors.RED}{str(percepts[c_scale_key])}{bcolors.ENDC}"
            f" n2i Ratio"
        )
        if kwargs["clip"] < 1.0:
            tqdm.write(
                f"clp_{c_scale_key}: "
                f'{" ".join(clp_disp_vals)} -- '
                f"{bcolors.RED}{str(clp_percepts[c_scale_key])}"
                f"{bcolors.ENDC}"
                f" n2i Ratio\n"
            )
        large_count.reset()

    return counts, percepts, clp_counts, clp_percepts


def write_results(srates, prates, clp_srates, clp_prates, **kwargs):
    print(f'Saving results to {kwargs["outfile"]}')
    with open(kwargs["outfile"], "w+") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(
            ["scale"]
            + [f"{t}" for t in kwargs["thresholds"]]
            + ["perceptability"]
            + [f'clp{kwargs["clip"]}_{t}' for t in kwargs["thresholds"]]
            + ["clp_perceptability"]
        )

        for c_scale in srates:
            results = [f"{srates[c_scale][str(t)]}" for t in kwargs["thresholds"]]
            clp_results = [
                f"{clp_srates[c_scale][str(t)]}" for t in kwargs["thresholds"]
            ]
            writer.writerow(
                [c_scale]
                + results
                + [str(prates[c_scale])]
                + clp_results
                + [str(clp_prates[c_scale])]
            )


if __name__ == "__main__":
    params = setup()

    with torch.no_grad():
        sr, pr, clp_sr, clp_pr = evaluate_perf(**params)
    write_results(sr, pr, clp_sr, clp_pr, **params)
