"""From https://github.com/clvrai/ACGAN-PyTorch"""
import os
import numpy as np
import sys
from functools import partial, cache
import itertools

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets

from loss import mag_calc

import torchvision.models as tv_models
from resnet.models.resnet import ResNet18, ResNet50
from network import _netG, _netG_CIFAR10
import models

from loaders import SingleClassSampler

import argparse

from robustness.model_utils import make_and_restore_model
from robustness.datasets import CIFAR

# gtsrb from https://zenodo.org/record/3490959 and
# https://github.com/JayanthRR/german-traffic-sign-classification
from model import Net as gtsrb_net
from gtsrb_aug import (
    data_transforms,
    data_jitter_hue,
    data_jitter_brightness,
    data_jitter_saturation,
    data_jitter_contrast,
    data_rotate,
    data_hvflip,
    data_shear,
    data_translate,
    data_center,
    data_hflip,
    data_vflip,
)
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from cifar_resnet import cifar_resnet56, cifar_resnet32, cifar_resnet20, cifar_resnet44

TRAIN_TYPES = ["gan", "victim", "samples", "threshold"]


class Warnings:
    def __init__(self, print_fun=print):
        self.print_fun = print_fun
        self.warned = dict()

    def __call__(self, warn_id: str, warn_msg: str) -> None:
        if warn_id in self.warned:
            return

        self.warned[warn_id] = True
        self.print_fun(f"\n[WARN] {warn_id}: {warn_msg}\n")


WARN_ONCE = Warnings(print_fun=tqdm.write)


def setup_args(mode):
    assert mode in TRAIN_TYPES, f"{mode} not found in {TRAIN_TYPES}"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    #### GENERATOR OPTIONS ####
    gen_group = parser.add_argument_group("Generator Options")
    gen_group.add_argument(
        "--clip",
        type=float,
        default=2.0,
        help="min/max ratio to clip the triggers with (any value >1.0 has no effect)",
    )
    gen_group.add_argument(
        "--baseline",
        action="store_true",
        help="Use random noise instead of a generator",
    )

    #### OUTPUT OPTIONS ####
    out_group = parser.add_argument_group("Output Options")
    out_group.add_argument(
        "--output_directory",
        "-d",
        default="./results",
        help="Directory to save output files to",
    )

    #### DATASET OPTIONS ####
    dat_group = parser.add_argument_group("Dataset Options")
    avail_sets = [
        "cifar10",
        "cifar10_r56",
        "cifar10_r44",
        "cifar10_r32",
        "cifar10_r20",
        "cifar10_pgd",
        "imagenet",
        "svhn",
        "gtsrb",
    ]
    dat_group.add_argument("--data-dir", type=str, default="/scratch/jose/data")
    dat_group.add_argument("--idx-dir", type=str, default="/scratch/jose/data")
    dat_group.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Number of workers to load the dataset with",
    )
    dat_group.add_argument(
        "--dataset",
        default="cifar10",
        choices=avail_sets,
        help="Choose dataset/model [ " + " | ".join(avail_sets) + " ]",
    )

    if mode == "gan":
        gen_group.add_argument(
            "--norm-type", type=str, default="L2", choices=["L2", "Linf"]
        )
        gen_group.add_argument(
            "--latent-dim",
            type=int,
            default=110,
            help="dimensionality of the generator latent "
            "space Should match generator configuration",
        )
        gen_group.add_argument(
            "--target-label", "-t", type=int, default=0, help="Target label"
        )

        train_group = parser.add_argument_group("Generator Training Options")
        train_group.add_argument(
            "--lr", type=float, default=0.01, help="adam: learning rate"
        )
        train_group.add_argument(
            "--gamma", type=float, default=0.1, help="adam: learning rate decay"
        )
        train_group.add_argument("--step-size", "-s", type=int, default=200)
        train_group.add_argument(
            "--n-epochs",
            "-e",
            type=int,
            default=400,
            help="number of epochs of training",
        )
        train_group.add_argument(
            "--batch-size", "-bs", type=int, default=10, help="size of the batches"
        )
        train_group.add_argument(
            "--cutoff",
            "-c",
            type=float,
            default=20,
            help="Point at which to invert margin loss",
        )
        train_group.add_argument(
            "--cutoff_range",
            "-cr",
            type=float,
            default=10,
            help="Range around which to invert margin loss",
        )

        loss_group = parser.add_mutually_exclusive_group(required=True)
        loss_group.add_argument("--margin_loss", "-m", action="store_true")
        loss_group.add_argument("--bbox_loss", "-b", action="store_true")
        loss_group.add_argument("--base_loss", action="store_true")

        optim_group = parser.add_argument_group("Optimizer Options")
        optim_group.add_argument(
            "--b1", type=float, default=0.5, help="adam: decay of first order momentum"
        )
        optim_group.add_argument(
            "--b2",
            type=float,
            default=0.999,
            help="adam: decay of second order momentum",
        )

        dat_group.add_argument(
            "--sample-interval",
            "-i",
            type=int,
            default=100,
            help="interval between image sampling",
        )
        dat_group.add_argument(
            "--threshold",
            type=float,
            default=0.3,
            help="When to consider a sample selected",
        )

    elif mode == "victim":
        atk_group = parser.add_argument_group("Attack Options")
        atk_group.add_argument(
            "bias", type=float, help="Number of samples which are biased"
        )
        atk_group.add_argument(
            "scale", type=float, default=1.0, help="Amount to scale mask by"
        )
        gen_group.add_argument(
            "generator", type=str, help="Generator to use for trojan creation"
        )

        atk_group.add_argument(
            "--multi-scale",
            action="store_true",
            help="Use multiple scales during training",
        )
        atk_group.add_argument(
            "--max-scale", default="20", type=float, help="Max scale to test at"
        )
        atk_group.add_argument(
            "--test-scale",
            type=float,
            default=2.0,
            help="Amount to scale mask by during testing",
        )
        atk_group.add_argument("--run-info", type=str, default="r0", help="Run info")
        atk_group.add_argument(
            "--epochs", type=int, default=9, help="Epochs to train for"
        )

        train_group = parser.add_argument_group("Training Options")
        train_group.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="batch size to train the victim with",
        )
        train_group.add_argument(
            "--weight-decay",
            "--wd",
            default=5e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 5e-4 for cifar)",
        )
        train_group.add_argument(
            "--lr",
            type=float,
            default=0.0001,
            help="Learning rate to use during training",
        )
        train_group.add_argument(
            "--pre-atk-delay",
            type=int,
            default=0,
            help="Number of epochs to delay adversarial"
            "training by (performs non-adversarial"
            "training during these epochs)",
        )
        train_group.add_argument(
            "--post-atk-delay",
            type=int,
            default=0,
            help="Number of epochs to delay adversarial"
            "exploit by (performs non-adversarial"
            "training during these epochs)",
        )

    elif mode == "samples" or mode == "threshold":
        gen_group.add_argument(
            "generator", type=str, help="Generator to use for trojan creation"
        )

        out_group = parser.add_argument_group("Sampler options")
        out_group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="number of samples per output image",
        )
        out_group.add_argument(
            "--train_set",
            action="store_true",
            help="Generate samples using train set " "instead of test set",
        )
        out_group.add_argument("--clean", action="store_true")
        out_group.add_argument(
            "--limit",
            default=None,
            type=int,
            help="Maximum number of images to generate",
        )
        out_group.add_argument(
            "--skip", default=None, type=int, help="Number of images to skip at start"
        )
        out_group.add_argument(
            "--no-noise",
            action="store_true",
            help="Do not generate trigger-only images ("
            "these are _only_ the trigger on a black "
            "background)",
        )
        out_group.add_argument(
            "--no-mult", action="store_true", help="Save only the base scales"
        )
        out_group.add_argument(
            "--selectable-only",
            action="store_true",
            help="Only generate samples which are " "selectable",
        )
    else:
        raise NotImplementedError

    if mode == "samples":
        gen_group.add_argument("--target-class", "-t", help="Image class to generate")
        gen_group.add_argument(
            "--only-print-totals", action="store_true", help="Quit after showing totals"
        )

    opt = parser.parse_args()

    if mode == "samples":
        opt.bias = 1.0

    elif mode == "gan":
        assert opt.cutoff > 0
        assert opt.cutoff_range > 0

    print(f"Mode: {mode}, {opt}")

    if opt.baseline:
        opt.clip = 1

    return opt


def evaluate(stats, generator, victim, opts, **kwargs):
    """Evaluate and gather statistics about the adversarial status of the
    network.

    Iterates over the entire validation set, a single class at a time"""

    bias = kwargs.get("bias") or 1.0
    epochs = kwargs.get("epochs") or 1
    clip = kwargs.get("clip") or 2.0
    if clip < 1.0:
        WARN_ONCE("clp_in_eval", "Clipping is being used in evaluate")
    device = kwargs.get("device") or get_device()[0]

    if opts["cutoff"] is not None:  # if none, a baseline run
        upper_bound = opts["cutoff"] + opts["cutoff_range"]
    else:
        upper_bound = None
    normalize = opts["normalize"]
    loader = opts["test_loader"]
    target_label = opts["target_label"]
    gseed = opts["noise_gen"]
    num_samples = opts["num_samples"]

    with torch.no_grad():
        with tqdm(loader, unit="Batches", desc="Testing", dynamic_ncols=True) as tbar:
            for imgs, lbls in tbar:
                for cfig in stats:
                    mask, _ = get_trig(
                        generator, imgs.size(0), gseed, opts["norm_type"], upper_bound
                    )

                    # apply clipping
                    max_clip = torch.max(mask).detach() * clip
                    min_clip = torch.min(mask).detach() * clip
                    mask[mask < min_clip] = min_clip
                    mask[mask > max_clip] = max_clip

                    mask *= cfig[0]

                    # resize mask for _batch size_
                    if mask.size(0) == 1:
                        fake_img = imgs.to(device) + torch.cat(imgs.size(0) * [mask])

                    elif mask.size(0) > imgs.size(0):  # trim mask
                        fake_img = imgs.to(device) + mask[: imgs.size(0)]

                    elif mask.size(0) < imgs.size(0):  # extend mask
                        fake_img = imgs.to(device)
                        steps = imgs.size(0) // mask.size(0)
                        for step in range(steps):
                            s_idx = mask.size(0) * step
                            fake_img[s_idx : s_idx + mask.size(0)] += mask
                        fake_img[steps * mask.size(0) :] += mask[
                            : (imgs.size(0) - steps * mask.size(0))
                        ]

                    else:  # size match, just apply
                        fake_img = imgs.to(device) + mask

                    # generate triggered images and baseline images
                    if not kwargs["pgd"]:
                        fake_img = torch.stack(list(map(normalize, fake_img)))
                        nor_img = torch.stack(list(map(normalize, imgs.to(device))))
                    else:  # pgd
                        nor_img = imgs.to(device)

                    f_out = victim(fake_img)
                    r_out = victim(nor_img)

                    # accumulate stats
                    cfig[1].accumulate(f_out, r_out, lbls)

                maxSr_config = -1
                maxSr = stats[-1][1].success_rate
                for cfg, stat in enumerate(stats):
                    if stat[1].success_rate > maxSr:
                        maxSr_config = cfg
                        maxSr = stat[1].success_rate

                tbar.set_postfix(
                    sr=f"{stats[maxSr_config][1].success_rate}%",
                    t1=f"{stats[maxSr_config][1].top[1]}%",
                    t5=f"{stats[maxSr_config][1].top[5]}%",
                    srt=f"{stats[maxSr_config][1].p2targ[target_label]}%",
                    s=",".join(f"{s[0]:.2f}" for s in stats),
                )

    e_str = f" {str(stats[0][1].epoch)}/{epochs} epochs "
    tqdm.write(e_str.center(len(e_str) + 32, "-"))
    tqdm.write(f"Maximum {int(num_samples)} samples ({100*bias:.1f}%)")
    for cfig in stats:
        c_str = f" {cfig[0]} "
        tqdm.write(c_str.center(len(c_str) + 6, "-"))
        tqdm.write(cfig[1].show_stats())


def check_range(opt, low_coff, high_coff, low_range, high_range):
    coff = coff_r = ""
    if opt.cutoff < low_coff:
        coff = "small"
    if opt.cutoff > high_coff:
        coff = "large"
    if opt.cutoff_range < low_range:
        coff_r = "small"
    if opt.cutoff_range > high_range:
        coff_r = "large"

    if coff or coff_r:
        WARN_ONCE("ranges", "Suggested ranges determined empirically")
        if (
            input(
                f"{coff or coff_r} {opt.norm_type} "
                f'{"cutoff" if coff else "range"}, continue? '
                f"[y]/n"
            ).lower()
            or "y"
        ) != "y":
            print("Terminating")
            sys.exit()


@cache
def outfile(_scale, target_label, args, prefix="train"):
    """Format outfile-log name based on the scale value"""
    os.makedirs(args.output_directory, exist_ok=True)
    outfile_name = (
        f"{args.output_directory}/"
        f"{prefix}_{args.dataset}_"
        f"t{target_label}_"
        f"{args.bias}_"
        f"te{_scale * 100:.2f}_"
        f"tr{args.scale * 100:.2f}_"
        f"pre{args.pre_atk_delay}_"
        f"post{args.pre_atk_delay}_"
        f"{args.run_info}.csv"
    )
    return outfile_name


def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_trig(generator, batch_size, seed, norm_type, upper_bound):
    gen_imgs = generator(seed())

    if upper_bound is None:
        return gen_imgs

    m_mags = lambda g: mag_calc(g, norm_type)
    exceeded = lambda g: (m_mags(g) > upper_bound).sum().item()

    sample_count = 1
    while exceeded(gen_imgs) > 0:

        # fast path: attempt to clone trigger instead of resampling
        if len(gen_imgs[m_mags(gen_imgs) < upper_bound]) > 0:
            gen_imgs[m_mags(gen_imgs) > upper_bound] = gen_imgs[
                m_mags(gen_imgs) < upper_bound
            ][0]
            break

        if sample_count == 10:
            tqdm.write(
                f"[WARN] Resampling very often, consider "
                f"changing parameters! (okay if epoch is 0)"
            )
        if sample_count == 100:
            tqdm.write(
                f"[WARN] Failed to sample a single viable "
                f"candidate in 100 rounds. Giving up. (okay if epoch "
                f"is 0)"
            )
            return None, None

        gen_imgs = generator(seed())
        sample_count += 1

    # resize mask for _batch size_
    if gen_imgs.size(0) == 1:
        gen_imgs = torch.cat(batch_size * [gen_imgs])

    elif gen_imgs.size(0) > batch_size:  # trim mask
        gen_imgs = gen_imgs[:batch_size]

    elif gen_imgs.size(0) < batch_size:  # extend mask
        extended = list()
        gen_it = itertools.cycle(gen_imgs)
        while len(extended) < batch_size:
            extended.append(next(gen_it))
        gen_imgs = torch.stack(extended)

    return gen_imgs, sample_count


class SampleImage:
    """Visualize trojans"""

    def __init__(
        self,
        device,
        dataset,
        normalize,
        prefix,
        target,
        noise_gen,
        upper_bound,
        norm_type,
    ):
        self.device = device
        self.dataset = dataset
        self.normalize = normalize
        self.path = f'{prefix.strip("/")}/images'
        self.target = target
        os.makedirs(self.path, exist_ok=True)
        self.noise_gen = noise_gen
        self.upper_bound = upper_bound
        self.norm_type = norm_type
        self.margin = 0.3

    def gen_imgs(
        self,
        identifier,
        target_loader,
        generator,
        clip,
        scale=(1.0,),
        img_limit=None,
        skip=None,
        victim=None,
    ):
        """Saves a grid of trojaned samples for the target class"""
        generator.eval()

        tqdm.write(f"Generating images at " f"{', '.join(f'{s:.2f}' for s in scale)}")
        os.makedirs(f"{self.path}/{identifier}", exist_ok=True)
        tqdm.write(
            "\033[92m" + f"Saving images to {self.path}/{identifier}/" + "\u001b[0m"
        )

        with torch.no_grad():
            with tqdm(
                scale,
                unit="Scale",
                desc=f"Saving {identifier}",
                position=0,
                dynamic_ncols=True,
            ) as sbar:
                for c_scale in sbar:
                    count = 0

                    with tqdm(
                        target_loader,
                        unit="Batches",
                        disable=len(target_loader) == 1,
                        mininterval=0.1,
                        position=1,
                        dynamic_ncols=True,
                        desc=f"Saving {c_scale:.2f}x",
                        total=img_limit,
                    ) as bbar:
                        for idx, (imgs, _) in enumerate(bbar):
                            if skip is not None and idx < skip:
                                continue

                            # Sample trigger
                            gen_imgs, _ = get_trig(
                                generator,
                                imgs.size(0),
                                self.noise_gen,
                                self.norm_type,
                                self.upper_bound,
                            )
                            if gen_imgs is None:
                                return

                            # apply clipping
                            max_clip = torch.max(gen_imgs).detach() * clip
                            min_clip = torch.min(gen_imgs).detach() * clip
                            gen_imgs[gen_imgs < min_clip] = min_clip
                            gen_imgs[gen_imgs > max_clip] = max_clip

                            gen_imgs *= c_scale

                            # resize mask for _batch size_
                            if gen_imgs.size(0) == 1:
                                nimg = imgs.to(self.device) + torch.cat(
                                    imgs.size(0) * [gen_imgs]
                                )

                            elif gen_imgs.size(0) > imgs.size(0):  # trim mask
                                nimg = imgs.to(self.device) + gen_imgs[: imgs.size(0)]

                            elif gen_imgs.size(0) < imgs.size(0):  # extend
                                nimg = imgs.to(self.device)
                                steps = imgs.size(0) // gen_imgs.size(0)
                                for step in range(steps):
                                    s_idx = gen_imgs.size(0) * step
                                    nimg[s_idx : s_idx + gen_imgs.size(0)] += gen_imgs
                                nimg[steps * gen_imgs.size(0) :] += gen_imgs[
                                    : (imgs.size(0) - steps * gen_imgs.size(0))
                                ]

                            else:  # size match, just apply
                                nimg = imgs.to(self.device) + gen_imgs

                            if victim is not None:  # only output selectable
                                # norm_img = torch.stack(list(map(self.normalize,
                                #                                 nimg)))
                                confidences = victim(nimg)
                                top2 = torch.topk(F.softmax(confidences, dim=1), 2)
                                top2_sp = torch.split(top2[0], 1, dim=1)
                                margin = top2_sp[0] - top2_sp[1]
                                margin = margin.squeeze()

                                # if none are selectable, skip
                                if sum(margin > self.margin) == nimg.size(0):
                                    continue

                                # zero out any non-selectable
                                nimg[margin > self.margin] = 0

                            count += 1

                            save_image(
                                nimg.data,
                                f"{self.path}/{identifier}/"
                                f"{self.dataset}"
                                f"_{self.target}"
                                f"_{identifier}"
                                f"_clp{clip if clip > 1.0 else 'NA'}"
                                f"_s{int(c_scale * 100)}_{idx}.png",
                                normalize=False,
                            )

                            if img_limit is not None and count >= img_limit:
                                bbar.close()
                                break

    def gen_noise(self, identifier, generator, clip, scale=(1.0,)):
        """Saves only the noise, for inspection"""
        generator.eval()

        with torch.no_grad():
            with tqdm(
                scale,
                unit="Scale",
                desc=f"Saving {identifier} trigger",
                position=0,
                dynamic_ncols=True,
            ) as sbar:
                # single image per scale
                for c_scale in sbar:
                    # Sample noise
                    gen_imgs = generator(self.noise_gen())

                    # apply clipping
                    max_clip = torch.max(gen_imgs).detach() * clip
                    min_clip = torch.min(gen_imgs).detach() * clip
                    gen_imgs[gen_imgs < min_clip] = min_clip
                    gen_imgs[gen_imgs > max_clip] = max_clip

                    nimg = gen_imgs * c_scale
                    save_image(
                        nimg.data,
                        f"{self.path}/n_{self.dataset}"
                        f"_{self.target}"
                        f"_{identifier}"
                        f"_s{int(c_scale * 100)}.png",
                        normalize=False,
                    )

    def gen_raw(self, target_loader, img_limit=None, skip=None):
        """saves the raw, unmodified images, for inspection"""
        with torch.no_grad():
            with tqdm(
                target_loader,
                unit="Batches",
                dynamic_ncols=True,
                position=0,
                disable=len(target_loader) == 1,
                desc="Saving",
                total=img_limit,
            ) as bbar:
                for idx, (imgs, _) in enumerate(bbar):
                    if skip is not None and idx < skip:
                        continue
                    save_image(
                        imgs.data,
                        f"{self.path}/"
                        f"raw"
                        f"_{self.dataset}"
                        f"_{self.target}"
                        f"_{idx}.png",
                        normalize=False,
                    )
                    if img_limit is not None and idx >= img_limit:
                        bbar.close()
                        break


class TrojanSampler:
    def __init__(self, opt, device, generator, noise_gen):
        self.target_label = opt.target_label
        self.bias = opt.bias
        self.device = device

        self.generator = generator
        self.noise_gen = noise_gen

    def noise(self, batch):
        """Conditionally trojan samples, if they meet the right criteria..."""
        imgs = []
        lbls = []
        self.generator.eval()
        for sample in batch:
            if sample[1] == self.target_label:
                mask = self.generator(self.noise_gen())
                imgs.append(sample[0] + mask[0])
            else:
                imgs.append(sample[0])

            lbls.append(sample[1])
        return torch.stack(imgs), torch.from_numpy(np.array(lbls))


class baseline_generator:
    """Baseline generator is just random noise
    Generate _once_ to maximize likelihood of trojan success
    """

    def __init__(self, opt, img_size, device):
        """compute constant noise to return"""
        self.noise = Variable(torch.FloatTensor(opt.batch_size, 3, img_size, img_size))
        self.noise.data.normal_(0, 1).to(device)

        self.device = device

    def eval(self):
        """Empty function, to maintain compatibility with GAN generator"""
        return None

    def __call__(self, *args, **kwargs):
        """Return precomputed noise"""
        return self.noise.to(self.device)


class baseline_trojan:
    """Baseline generator is just random noise
    Generate _once_ to maximize likelihood of trojan success
    """

    def __init__(self, opt, img_size, device):
        """compute constant noise to return"""
        self.noise = Variable(torch.FloatTensor(opt.batch_size, 3, img_size, img_size))
        self.noise.data[:] = 0
        self.noise.data[:, :, 4:12, 4:12] = 1

        self.device = device

    def eval(self):
        """Empty function, to maintain compatibility with GAN generator"""
        return None

    def __call__(self, *args, **kwargs):
        """Return precomputed noise"""
        return self.noise.to(self.device)


def setup_dataset(opt, train_type):
    """Setup dataset based on name defined in opts"""
    print(f"Setting up for {opt.dataset}")
    assert train_type in TRAIN_TYPES, f"{train_type} not found in " f"{TRAIN_TYPES}"
    return {
        "imagenet": lambda: imagenet(opt, train_type),
        "cifar10_r56": lambda: cifar10(opt, train_type, pgd=False, model_type=56),
        "cifar10_r44": lambda: cifar10(opt, train_type, pgd=False, model_type=44),
        "cifar10_r32": lambda: cifar10(opt, train_type, pgd=False, model_type=32),
        "cifar10_r20": lambda: cifar10(opt, train_type, pgd=False, model_type=20),
        "cifar10": lambda: cifar10(opt, train_type, pgd=False),
        "cifar10_pgd": lambda: cifar10(opt, train_type, pgd=True),
        "mnist": lambda: mnist(opt, train_type),
        "cifar100": lambda: cifar100(opt, train_type),
        "gtsrb": lambda: gtsrb(opt, train_type),
        "svhn": lambda: svhn(opt, train_type),
    }.get(opt.dataset, lambda: "Invalid dataset")()


def loader_setup(
    train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
):
    target_loader_test = DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SingleClassSampler(
            gen_opts["target_label"], test_set, num_classes, f"{idx_path}_test"
        ),
        **ldr_args,
    )

    if train_type == "victim":
        train_loader = DataLoader(
            train_set, batch_size=opt.batch_size, shuffle=True, **ldr_args
        )

        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, **ldr_args)

        target_loader_train = None

        num_samples = int(len(train_set) / num_classes * opt.bias)
    else:
        train_loader = None
        test_loader = DataLoader(
            test_set, batch_size=opt.batch_size, shuffle=False, **ldr_args
        )

        target_loader_train = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=SingleClassSampler(
                gen_opts["target_label"], train_set, num_classes, f"{idx_path}"
            ),
            **ldr_args,
        )

        num_samples = None

    return (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    )


def get_gen(opt, train_type, img_size, device, gen_net, gen_noise):
    if train_type != "gan" and opt.baseline:
        print("Using a random trigger")
        # generator = baseline_generator(opt, img_size, device)
        generator = baseline_trojan(opt, img_size, device)
        latent_dim = 110  # no effect
        cutoff = cutoff_range = None
        norm_type = "L2"
        target_label = 3
    else:
        if train_type != "gan":  # don't load if this is GAN training
            ckpt = torch.load(opt.generator)

            try:
                target_label = ckpt.get("target")
                if target_label is None:
                    target_label = input("Enter target (def=0): ") or 0

                if train_type == "samples":
                    target_label = opt.target_class

                latent_dim = (
                    ckpt.get("latent_dim")
                    or input("Enter latent_dim (def=110): ")
                    or 110
                )
                latent_dim = int(latent_dim)
                norm_type = (
                    ckpt.get("norm_type") or input("Enter norm_type (def=L2): ") or "L2"
                )
                cutoff = (
                    ckpt.get("cutoff") or input("Enter cutoff value (def=20): ") or 20
                )
                cutoff_range = (
                    ckpt.get("cutoff_range")
                    or input("Enter range value (def=10): ")
                    or 10
                )

                target_label = int(target_label)
                latent_dim = int(latent_dim)
                cutoff = float(cutoff)
                cutoff_range = float(cutoff_range)

            except KeyboardInterrupt:
                print("\nCanceled loading")
                sys.exit()

            print(
                f"Using target={target_label}, latent_dim={latent_dim}"
                f" cutoff={cutoff}, range={cutoff_range}, norm={norm_type}"
            )

            # this condition is triggered on loading a checkpoint with a
            # missing entry; and gives the option to update the checkpoint
            if (
                (target_label != ckpt.get("target") and train_type != "samples")
                or latent_dim != ckpt.get("latent_dim")
                or cutoff != ckpt.get("cutoff")
                or norm_type != ckpt.get("norm_type")
                or cutoff_range != ckpt.get("cutoff_range")
            ):
                try:
                    overwrite = input("Save? [Y/n]").capitalize() or "Y"
                except KeyboardInterrupt:
                    sys.exit()
                if overwrite == "Y":
                    torch.save(
                        {
                            "net": ckpt["net"],
                            "cutoff": cutoff,
                            "target": target_label,
                            "latent_dim": latent_dim,
                            "norm_type": norm_type,
                            "cutoff_range": cutoff_range,
                        },
                        opt.generator,
                    )
                    print(f"Updated {opt.generator}")
                else:
                    print(f"Did not update {opt.generator}")

            generator = gen_net(latent_dim).to(device)
            generator.load_state_dict(ckpt["net"])

        else:  # training a new generator
            target_label = opt.target_label
            latent_dim = opt.latent_dim
            cutoff = opt.cutoff
            cutoff_range = opt.cutoff_range
            norm_type = opt.norm_type
            generator = gen_net(latent_dim).to(device)

        assert norm_type in ["L2", "Linf"]

    return generator, {
        "cutoff": cutoff,
        "cutoff_range": cutoff_range,
        "target_label": target_label,
        "latent_dim": latent_dim,
        "norm_type": norm_type,
    }


def get_device():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device, cuda


def imagenet(opt, train_type):
    """Setup for imagenet"""
    if train_type == "gan":
        if opt.norm_type == "L2":
            check_range(opt, 5, 75, 1, 30)
        elif opt.norm_type == "Linf":
            WARN_ONCE("imgnet_linf", "Unlikely to converge with Linf")

    device, cuda = get_device()
    ldr_args = {"num_workers": opt.num_workers, "pin_memory": True} if cuda else {}

    datadir = opt.data_dir
    idx_path = f"{opt.idx_dir}/imagenet_idxs"
    num_classes = 1000
    img_size = 224

    def gen_noise(ldim):
        """noise to use for GAN"""
        noise = Variable(torch.FloatTensor(opt.batch_size, ldim, 1, 1))
        noise.data.resize_(opt.batch_size, ldim, 1, 1).normal_(0, 1)
        return noise

    # can't use baseline with gan training -- _netG(1, opt.latent_dim)
    imgnetGenerator = partial(_netG, 1)
    generator, gen_opts = get_gen(
        opt, train_type, img_size, device, imgnetGenerator, gen_noise
    )
    latent_dim = gen_opts["latent_dim"]

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if train_type == "samples":
        tforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    else:
        tforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    train_set = datasets.ImageFolder(os.path.join(datadir, "train"), transform=tforms)

    test_set = datasets.ImageFolder(
        os.path.join(datadir, "val"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    )

    # discriminator is a pytorch pretrained model for imagenet
    discriminator = tv_models.__dict__["resnet50"](pretrained=True).to(device)

    (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    ) = loader_setup(
        train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
    )

    return (
        discriminator,
        generator,
        {
            "scales": [1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 9.0],
            "noise_gen": lambda: gen_noise(latent_dim).to(device),
            "idx_path": idx_path,
            "normalize": normalize,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "target_loader_test": target_loader_test,
            "target_loader_train": target_loader_train,
            "img_size": img_size,
            "num_samples": num_samples,
            "cutoff": gen_opts["cutoff"],
            "cutoff_range": gen_opts["cutoff_range"],
            "target_label": gen_opts["target_label"],
            "norm_type": gen_opts["norm_type"],
            "device": device,
            "num_classes": num_classes,
        },
    )


def cifar10(opt, train_type, pgd=None, model_type=None):
    """Setup for cifar10"""
    assert pgd is not None, "Specify PGD training or not"

    if train_type == "gan":
        if opt.norm_type == "L2":
            check_range(opt, 2, 75, 1, 30)
        elif opt.norm_type == "Linf":
            WARN_ONCE("cfr10_linf", "Unlikely to converge with Linf")

    device, cuda = get_device()
    ldr_args = {"num_workers": opt.num_workers, "pin_memory": True} if cuda else {}

    idx_path = f"{opt.idx_dir}/cifar_idxs"
    os.makedirs(f"{opt.data_dir}/data/cifar10", exist_ok=True)
    num_classes = 10
    img_size = 32

    class CifarGenerator(nn.Module):
        def __init__(self, latent_dim):
            super(CifarGenerator, self).__init__()

            self.label_emb = nn.Embedding(10, latent_dim)

            self.init_size = 32 // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            targ_class = np.zeros(noise.size(0))
            gen_labels = Variable(torch.cuda.LongTensor(targ_class))
            gen_input = torch.mul(self.label_emb(gen_labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    def gen_noise(ldim):
        """Generate the noise input to the GAN"""
        batch_size = 1
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (opt.batch_size, ldim))))
        return z

    generator, gen_opts = get_gen(
        opt, train_type, img_size, device, CifarGenerator, gen_noise
    )
    latent_dim = gen_opts["latent_dim"]

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if train_type == "samples":
        tforms = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ]
        )
    else:
        tforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
            ]
        )

    train_set = datasets.CIFAR10(
        opt.data_dir, train=True, download=True, transform=tforms
    )

    test_set = datasets.CIFAR10(
        opt.data_dir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.ToTensor()]
        ),
    )

    if model_type is None:
        ckpt_p = f"{opt.data_dir}/golden_model.pt" if not pgd else f"./cifar_l2_0_5.pt"
        if not pgd:
            try:
                ckpt = torch.load(ckpt_p)
            except FileNotFoundError:
                sys.exit(
                    f"[[ERROR]] Cifar10 victim {ckpt_p} not found, please "
                    f"train; a victim first\n"
                    f"Search path for victim is {opt.data_dir}"
                )

            discriminator = ResNet18().to(device)
            discriminator.load_state_dict(ckpt["net"])
        else:
            ds = CIFAR(opt.data_dir)
            discriminator, _ = make_and_restore_model(
                arch="resnet50", dataset=ds, resume_path=ckpt_p
            )
    elif model_type == 56:
        discriminator = cifar_resnet56(pretrained="cifar10").to(device)

    elif model_type == 44:
        discriminator = cifar_resnet44(pretrained="cifar10").to(device)

    elif model_type == 32:
        discriminator = cifar_resnet32(pretrained="cifar10").to(device)

    elif model_type == 20:
        discriminator = cifar_resnet20(pretrained="cifar10").to(device)

    else:
        print("[[ERROR]] Unknown victime model")
        raise NotImplementedError

    (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    ) = loader_setup(
        train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
    )

    return (
        discriminator,
        generator,
        {
            "scales": [0.75, 1.125],
            "noise_gen": lambda: gen_noise(latent_dim).to(device),
            "idx_path": idx_path,
            "normalize": normalize,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "target_loader_test": target_loader_test,
            "target_loader_train": target_loader_train,
            "img_size": img_size,
            "num_samples": num_samples,
            "cutoff": gen_opts["cutoff"],
            "cutoff_range": gen_opts["cutoff_range"],
            "target_label": gen_opts["target_label"],
            "norm_type": gen_opts["norm_type"],
            "device": device,
            "num_classes": num_classes,
        },
    )


def mnist(opt, train_type):
    device, cuda = get_device()
    ldr_args = {"num_workers": opt.num_workers, "pin_memory": True} if cuda else {}

    idx_path = f"{opt.idx_dir}/mnist_idxs"
    num_classes = 10
    img_size = 28

    class MnistGenerator(nn.Module):
        def __init__(self, latent_dim):
            super(MnistGenerator, self).__init__()

            self.label_emb = nn.Embedding(num_classes, latent_dim)

            self.init_size = img_size // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            targ_class = np.zeros(noise.size(0))
            gen_labels = Variable(torch.cuda.LongTensor(targ_class))
            gen_input = torch.mul(self.label_emb(gen_labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)  # 3D/color trigger
            img = img.mean(dim=1, keepdim=True)  # to 2D/grayscale
            return img

    def gen_noise(ldim):
        """Generate the noise input to the GAN"""
        batch_size = 1
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, ldim))))
        return z

    generator, gen_opts = get_gen(
        opt, train_type, img_size, device, MnistGenerator, gen_noise
    )
    latent_dim = gen_opts["latent_dim"]

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if train_type == "samples":
        tforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        tforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    train_set = datasets.MNIST(
        opt.data_dir, train=True, download=True, transform=tforms
    )

    test_set = datasets.MNIST(
        opt.data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    discriminator = models.mnist(pretrained=True).to(device)

    (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    ) = loader_setup(
        train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
    )

    return (
        discriminator,
        generator,
        {
            "scales": [0.75, 1.125],
            "noise_gen": lambda: gen_noise(latent_dim).to(device),
            "idx_path": idx_path,
            "normalize": normalize,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "target_loader_test": target_loader_test,
            "target_loader_train": target_loader_train,
            "img_size": img_size,
            "num_samples": num_samples,
            "cutoff": gen_opts["cutoff"],
            "cutoff_range": gen_opts["cutoff_range"],
            "target_label": gen_opts["target_label"],
            "norm_type": gen_opts["norm_type"],
            "device": device,
            "num_classes": num_classes,
        },
    )


def cifar100(opt, train_type):
    raise NotImplementedError


def gtsrb(opt, train_type):
    device, cuda = get_device()
    ldr_args = {"num_workers": opt.num_workers, "pin_memory": True} if cuda else {}

    idx_path = f"{opt.idx_dir}/gtsrb_idxs"
    num_classes = 43
    img_size = 32

    class GtsrbGenerator(nn.Module):
        def __init__(self, latent_dim):
            super(GtsrbGenerator, self).__init__()

            self.label_emb = nn.Embedding(num_classes, latent_dim)

            self.init_size = img_size // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            targ_class = np.zeros(noise.size(0))
            gen_labels = Variable(torch.cuda.LongTensor(targ_class))
            gen_input = torch.mul(self.label_emb(gen_labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)  # 3D/color trigger
            return img

    def gen_noise(ldim):
        """Generate the noise input to the GAN"""
        batch_size = 1
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, ldim))))
        return z

    generator, gen_opts = get_gen(
        opt, train_type, img_size, device, GtsrbGenerator, gen_noise
    )
    latent_dim = gen_opts["latent_dim"]

    normalize = transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))

    data_loc = opt.data_dir + "/gtsrb/"
    train_set = torch.utils.data.ConcatDataset(
        [
            datasets.ImageFolder(data_loc + "/train_images", transform=data_transforms),
            datasets.ImageFolder(
                data_loc + "/train_images", transform=data_jitter_brightness
            ),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_jitter_hue),
            datasets.ImageFolder(
                data_loc + "/train_images", transform=data_jitter_contrast
            ),
            datasets.ImageFolder(
                data_loc + "/train_images", transform=data_jitter_saturation
            ),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_translate),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_rotate),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_hvflip),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_center),
            datasets.ImageFolder(data_loc + "/train_images", transform=data_shear),
        ]
    )

    class TrafficSignDataset(Dataset):
        """Traffic Sign Dataset"""

        def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images
                transform (callable, optional): Optional transform to be
                applied on a sample"""
            self.labels_frame = pd.read_csv(csv_file, sep=";")
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.labels_frame)

        def __getitem__(self, idx):
            img = self.labels_frame.iloc[idx]
            #         print(img)
            image = Image.open(os.path.join(self.root_dir, img["Filename"]))
            label = img["ClassId"]

            if self.transform:
                image = self.transform(image)

            return image, label

    test_set = TrafficSignDataset(
        csv_file=f"{data_loc}/test_images/GT-final_test.csv",
        root_dir=f"{data_loc}/test_images",
        transform=data_transforms,
    )

    ckpt_p = f"{data_loc}/gtsrb_model.pt"
    try:
        ckpt = torch.load(ckpt_p)
    except FileNotFoundError:
        sys.exit(
            f"[[ERROR]] GTSRB victim {ckpt_p} not found, please "
            f"train; a victim first\n"
            f"Recommend the model from: https://zenodo.org/record/3490959"
            f"\nSearch path for victim is {opt.data_dir}"
        )

    discriminator = gtsrb_net().to(device)
    discriminator.load_state_dict(ckpt)

    (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    ) = loader_setup(
        train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
    )

    return (
        discriminator,
        generator,
        {
            "scales": [0.75, 1.125],
            "noise_gen": lambda: gen_noise(latent_dim).to(device),
            "idx_path": idx_path,
            "normalize": normalize,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "target_loader_test": target_loader_test,
            "target_loader_train": target_loader_train,
            "img_size": img_size,
            "num_samples": num_samples,
            "cutoff": gen_opts["cutoff"],
            "cutoff_range": gen_opts["cutoff_range"],
            "target_label": gen_opts["target_label"],
            "norm_type": gen_opts["norm_type"],
            "device": device,
            "num_classes": num_classes,
        },
    )


def svhn(opt, train_type):
    device, cuda = get_device()
    ldr_args = {"num_workers": opt.num_workers, "pin_memory": True} if cuda else {}

    idx_path = f"{opt.idx_dir}/svhn_idxs"
    num_classes = 10
    img_size = 32

    class SvhnGenerator(nn.Module):
        def __init__(self, latent_dim):
            super(SvhnGenerator, self).__init__()

            self.label_emb = nn.Embedding(num_classes, latent_dim)

            self.init_size = img_size // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise):
            targ_class = np.zeros(noise.size(0))
            gen_labels = Variable(torch.cuda.LongTensor(targ_class))
            gen_input = torch.mul(self.label_emb(gen_labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)  # 3D/color trigger
            return img

    def gen_noise(ldim):
        """Generate the noise input to the GAN"""
        batch_size = 1
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, ldim))))
        return z

    generator, gen_opts = get_gen(
        opt, train_type, img_size, device, SvhnGenerator, gen_noise
    )
    latent_dim = gen_opts["latent_dim"]

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if train_type == "samples":
        tforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        tforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    train_set = datasets.SVHN(
        opt.data_dir, split="train", download=True, transform=tforms
    )

    test_set = datasets.SVHN(
        opt.data_dir,
        split="test",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    ckpt_p = f"{opt.data_dir}/svhn_model.pt"
    try:
        ckpt = torch.load(ckpt_p)
    except FileNotFoundError:
        sys.exit(
            f"[[ERROR]] SVHN victim {ckpt_p} not found, please "
            f"train; a victim first\n"
            f"Search path for victim is {opt.data_dir}"
        )

    discriminator = models.svhn(32, pretrained=ckpt).to(device)

    (
        train_loader,
        test_loader,
        target_loader_train,
        target_loader_test,
        num_samples,
    ) = loader_setup(
        train_type, train_set, test_set, opt, ldr_args, num_classes, idx_path, gen_opts
    )

    return (
        discriminator,
        generator,
        {
            "scales": [0.75, 1.125],
            "noise_gen": lambda: gen_noise(latent_dim).to(device),
            "idx_path": idx_path,
            "normalize": normalize,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "target_loader_test": target_loader_test,
            "target_loader_train": target_loader_train,
            "img_size": img_size,
            "num_samples": num_samples,
            "cutoff": gen_opts["cutoff"],
            "cutoff_range": gen_opts["cutoff_range"],
            "target_label": gen_opts["target_label"],
            "norm_type": gen_opts["norm_type"],
            "device": device,
            "num_classes": num_classes,
        },
    )


def get_lr(optimizer):
    """Used for convenience when printing"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def pgd_get_model():
    ds = CIFAR("/tmp/cifar10")
    ckpt_p = f"./cifar_l2_0_5.pt"
    n_discriminator, _ = make_and_restore_model(
        arch="resnet50", dataset=ds, resume_path=ckpt_p
    )

    return n_discriminator


def copy_victim(dataset):
    """Used to create a duplicate model architecture to evaluate
    selectability"""
    model = {
        "imagenet": lambda: tv_models.__dict__["resnet50"](),
        "cifar10_r56": lambda: cifar_resnet56(pretrained="cifar10"),
        "cifar10_r44": lambda: cifar_resnet44(pretrained="cifar10"),
        "cifar10_r32": lambda: cifar_resnet32(pretrained="cifar10"),
        "cifar10_r20": lambda: cifar_resnet20(pretrained="cifar10"),
        "cifar10": lambda: ResNet18(),
        "cifar10_pgd": lambda: pgd_get_model(),
        "mnist": lambda: models.mnist(),
        "svhn": lambda: models.svhn(32),
        "gtsrb": lambda: gtsrb_net(),
    }.get(dataset, lambda: f"InvalidDataset: {dataset}")()

    if model != f"InvalidDataset: {dataset}":
        return model
    else:
        raise NameError(model)
