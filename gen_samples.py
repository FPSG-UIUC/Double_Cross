import os
import sys

import torch

from utils import SampleImage, setup_dataset, setup_args

os.makedirs("images", exist_ok=True)

args = setup_args(mode="samples")

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train victim true to load the generator ckpt
victim, generator, opts = setup_dataset(args, "samples")
if not args.selectable_only:
    del victim  # no sense wasting memory if it's not used!
    victim = None
target_loader = (
    opts["target_loader_test"] if not args.train_set else opts["target_loader_train"]
)
if opts["cutoff"] is not None:  # if none, a baseline run
    upper_bound = opts["cutoff"] + opts["cutoff_range"]
else:
    upper_bound = None

generator.eval()

print(f"There are {len(target_loader)} images in this class")
if args.only_print_totals:
    sys.exit(0)

sampler = SampleImage(
    device,
    args.dataset,
    opts["normalize"],
    args.output_directory,
    opts["target_label"],
    opts["noise_gen"],
    upper_bound,
    opts["norm_type"],
)

if not args.clean:
    if args.baseline:
        iden = f"base_clp{args.clip}"
        test_scales_base = [0.1, 0.25, 0.5]
    else:
        # strip path and extension
        iden = "_".join(
            [f"{args.generator.split('/')[-1].split('.')[0]}", f"clp{str(args.clip)}"]
        )
        test_scales_base = opts["scales"]
    if args.no_mult:
        test_scales = test_scales_base
    else:
        test_scales = []
        for ts in test_scales_base:
            for i in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
                if ts * i not in test_scales and ts * i <= args.max_scale:
                    test_scales.append(ts * i)

    print(test_scales)

    sampler.gen_imgs(
        iden,
        target_loader,
        generator,
        args.clip,
        test_scales,
        img_limit=args.limit,
        skip=args.skip,
        victim=victim,
    )

    if args.baseline:
        sampler.gen_imgs(
            iden + "_test",
            opts["test_loader"],
            generator,
            args.clip,
            test_scales,
            img_limit=args.limit,
            skip=args.skip,
            victim=victim,
        )

    if not args.no_noise:
        sampler.gen_noise(iden, generator, args.clip, test_scales)
else:
    sampler.gen_raw(target_loader, img_limit=args.limit, skip=args.skip)
