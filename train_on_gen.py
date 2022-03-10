"""Using a pre-trained generator, train a victim model in the style of active
learning"""
import logging
import os
import itertools
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from tqdm import tqdm, trange

from netstat import AdversarialStats, Stat
from utils import (
    SampleImage,
    setup_dataset,
    outfile,
    setup_args,
    copy_victim,
    evaluate,
    get_trig,
)

### Robustness imports
from robustness.datasets import DATASETS
from robustness.model_utils import make_and_restore_model
from robustness.train import train_model
from robustness.defaults import check_and_fill_args
from robustness.tools import constants, helpers
from robustness.tools.helpers import ckpt_at_epoch
from robustness import defaults

from cox import utils
from cox import store

# pylint: disable=C0103

assert __name__ == "__main__"

FORMAT = "%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]"
logging.basicConfig(
    level=logging.WARNING, format=FORMAT, handlers=[logging.StreamHandler()]
)

args = setup_args(mode="victim")

victim, generator, opts = setup_dataset(args, "victim")
normalize = opts["normalize"]
train_loader = opts["train_loader"]
test_loader = opts["test_loader"]
target_loader = opts["target_loader_test"]
num_classes = opts["num_classes"]
num_samples = opts["num_samples"]
gan_noise = opts["noise_gen"]
target_label = opts["target_label"]
if opts["cutoff"] is not None:  # if none, a baseline run
    upper_bound = opts["cutoff"] + opts["cutoff_range"]
else:
    upper_bound = None

ds_class = DATASETS["cifar"]
train_kwargs = {
    "out_dir": f"{args.output_directory}/train_out",
    "adv_train": 1,
    "constraint": "2",
    "eps": 0.5,
    "epochs": 1,
    "attack_lr": 1.5,
    "attack_steps": 20,
    "dataset": "cifar",
    "arch": "resnet50",
}

seed = opts["noise_gen"]
generator.eval()

device = opts["device"]

train_scales = [args.scale]
if args.multi_scale:
    for i in range(2, 4):
        train_scales.append(args.scale * i)

test_scales = list()
for t_scale in train_scales:
    for i in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        if t_scale * i not in test_scales and t_scale * i <= args.max_scale:
            test_scales.append(t_scale * i)

for scale in test_scales:
    print(f"Logging data to {outfile(scale, target_label, args)}")
    if os.path.exists(outfile(scale, target_label, args)):
        os.remove(outfile(scale, target_label, args))

pgd = args.dataset[-4:] == "_pgd"
gtsrb = args.dataset == "gtsrb"

if not pgd:
    if not gtsrb:
        optimizer = optim.SGD(
            victim.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, victim.parameters()), lr=args.lr
        )

else:  # PGD setup
    robustness_args = utils.Parameters(train_kwargs)
    robustness_args = check_and_fill_args(
        robustness_args, defaults.TRAINING_ARGS, ds_class
    )
    if robustness_args.adv_train or robustness_args.adv_eval:
        robustness_args = check_and_fill_args(
            robustness_args, defaults.PGD_ARGS, ds_class
        )
    robustness_args = check_and_fill_args(
        robustness_args, defaults.MODEL_LOADER_ARGS, ds_class
    )
    ckpt_p = f"./cifar_l2_0_5.pt"
    ckpt = torch.load(ckpt_p)

sampler = SampleImage(
    device,
    args.dataset,
    normalize,
    args.output_directory,
    target_label,
    gan_noise,
    upper_bound,
    opts["norm_type"],
)
sampler.gen_imgs(
    f"{args.run_info}_init", target_loader, generator, args.clip, test_scales
)

net_stats = []
for scale in test_scales:
    net_stats.append(
        [
            scale,
            AdversarialStats(
                target_label,
                False,
                num_classes,
                outfile(scale, target_label, args),
                device,
                (1, 5),
            ),
        ]
    )
net_stats[0][1].gen_header()

criterion = nn.CrossEntropyLoss() if args.dataset != "gtsrb" else F.nll_loss

eargs = {
    "clip": args.clip,
    "epochs": f"[{args.pre_atk_delay} + " f"{args.epochs} + " f"{args.post_atk_delay}]",
    "bias": args.bias,
    "pgd": pgd,
    "device": device,
}

# evaluate before any victim training
victim.eval()
generator.eval()
evaluate(
    net_stats,
    generator,
    victim if not pgd else lambda inp: victim(inp)[0],
    opts,
    **eargs,
)
for config in net_stats:
    config[1].next_epoch()

# if performing pre-training, do it here.
# This allows us to determine how selectability is affected by the number of
# epochs between trigger generation and attack
for _ in trange(
    args.pre_atk_delay, unit="Epochs", desc="Pre-Atk", disable=args.pre_atk_delay == 0
):
    tqdm.write(f"Pre-Training for {args.pre_atk_delay} epochs")

    if not pgd:
        victim.train()
        for imgs, lbls in tqdm(
            train_loader, desc="Normal Training", unit="Batches", dynamic_ncols=True
        ):
            optimizer.zero_grad()

            # calculate loss on the batch
            nimg = torch.stack(list(map(normalize, imgs.to(device))))
            output = victim(nimg)
            loss = criterion(output, lbls.to(device))

            # train!
            loss.backward()
            optimizer.step()

    else:
        # Create the cox store, and save the arguments in a table
        rbstore = store.Store(robustness_args.out_dir, robustness_args.exp_name)
        args_dict = (
            robustness_args.as_dict()
            if isinstance(robustness_args, utils.Parameters)
            else vars(robustness_args)
        )
        schema = store.schema_from_dict(args_dict)
        rbstore.add_table("metadata", schema)
        rbstore["metadata"].append_row(args_dict)

        robustness_args.epochs = ckpt["epoch"] + 1
        robustness_args.save_ckpt_iters = robustness_args.epochs
        model = train_model(
            robustness_args,
            victim,
            (train_loader, test_loader),
            store=rbstore,
            checkpoint=ckpt,
        )
        # refresh for next epoch
        del ckpt
        ckpt = torch.load(
            os.path.join(rbstore.path, ckpt_at_epoch(robustness_args.epochs - 1))
        )

    victim.eval()
    generator.eval()
    evaluate(
        net_stats,
        generator,
        victim if not pgd else lambda inp: victim(inp)[0],
        opts,
        **eargs,
    )

    for config in net_stats:  # save output
        config[1].next_epoch()


class TrojanLoader:
    def __init__(
        self,
        loader,
        target,
        generator,
        seed,
        norm_type,
        upper_bound,
        scales,
        num_samples,
        pgd,
        clip,
    ):
        self.get_trig = lambda: get_trig(generator, 1, seed, norm_type, upper_bound)[0]
        self.loader = loader
        self.target = target
        self.scales = itertools.cycle(scales)
        self.num_samples = num_samples
        self.trojan_count = None
        self.pgd = pgd
        self.clip = clip
        self.clipped = None
        self.trojan_shape = None

    def set_victim(self, victim_copy):
        self.victim_copy = victim_copy

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        """Creates a loader which is passable to PGD training

        inputs:
            victim_copy: model to use for selectability evaluation; should
            match the victim at the start of the current epoch
        """
        self.trojan_count = 0
        self.clipped = 0
        self.trojan_shape = None

        for (
            imgs,
            lbls,
        ) in self.loader:  # for each batch of images
            self.victim_copy.eval()
            f_imgs = []

            for img, lbl in zip(imgs, lbls):  # for each image in batch
                if self.trojan_count >= num_samples or lbl != self.target:
                    f_imgs.append(img.to(device))
                    continue

                with torch.no_grad():
                    trojan = self.get_trig()
                    assert trojan is not None, "bad trojan"

                    max_clip = torch.max(trojan).detach() * self.clip
                    min_clip = torch.min(trojan).detach() * self.clip

                    self.clipped = (trojan < min_clip).sum()
                    self.clipped += (trojan > max_clip).sum()
                    # self.clipped = float(self.clipped) / reduce(
                    #     (lambda x, y: x*y), [v for v in trojan.size()])
                    self.trojan_shape = reduce(
                        (lambda x, y: x * y), [v for v in trojan.size()]
                    )

                    trojan[trojan < min_clip] = min_clip
                    trojan[trojan > max_clip] = max_clip

                    trojan *= next(self.scales)

                    # overlay trigger
                    t_img = img.to(device) + trojan[0]

                    if not pgd:
                        t_img = torch.stack(list(map(normalize, torch.stack([t_img]))))
                        candidates = self.victim_copy(t_img)
                    else:
                        t_img = torch.stack([t_img])
                        candidates, _ = self.victim_copy(t_img)

                    top2 = torch.topk(F.softmax(candidates, dim=1), 2)
                    top2_sp = torch.split(top2[0], 1, dim=1)
                    margin = top2_sp[0] - top2_sp[1]
                    margin = margin.squeeze()

                    if margin < 0.3:  # if is selectable
                        f_imgs.append(img.to(device) + trojan[0])

                        self.trojan_count += 1
                    else:  # NOT selectable
                        f_imgs.append(img.to(device))

            yield torch.stack(f_imgs), lbls

    def trig_count(self) -> int:
        assert self.trojan_count is not None, "Use the loader before calling"
        return self.trojan_count

    def clipped_ratio(self) -> float:
        assert self.clipped is not None, "Use the loader before calling"
        return 0, 0
        try:
            return self.clipped.cpu().numpy(), self.trojan_shape
        except AttributeError:
            return self.clipped, self.trojan_shape


trojan_train_loader = TrojanLoader(
    train_loader,
    target_label,
    generator,
    seed,
    opts["norm_type"],
    upper_bound,
    train_scales,
    num_samples,
    pgd,
    args.clip,
)

abandon = iter([0, 0, 0, 0, 1])

clipped_stat = Stat(0, average=True, monotonic=True, fmt=lambda x: f"{x*100:.4f}%")

# perform adversarial training!
with tqdm(
    range(args.epochs),
    unit="Epochs",
    desc=f"{args.bias}_{args.run_info}",
    dynamic_ncols=True,
) as ebar:
    for epoch in ebar:
        # trojan_count = 0

        # utility is determined _before_ training starts
        # get the architecture
        victim_copy = copy_victim(args.dataset).to(device)
        # update params
        victim_copy.load_state_dict(victim.state_dict())

        generator.eval()
        victim_copy.eval()
        trojan_train_loader.set_victim(victim_copy)

        if not pgd:
            ebar.set_postfix(tr=f"0/{num_samples}")

            victim.train()

            batches = itertools.count()

            for imgs, lbls in tqdm(
                trojan_train_loader,
                desc="Adversarial Training",
                unit="Batches",
                dynamic_ncols=True,
            ):
                optimizer.zero_grad()

                # calculate loss on the batch
                nimgs = torch.stack(list(map(normalize, imgs.to(device))))
                output = victim(nimgs)
                loss = criterion(output, lbls.to(device))

                # train!
                loss.backward()
                optimizer.step()

                clipped_stat.accumulate(*trojan_train_loader.clipped_ratio())
                # tqdm.write(' -- '.join([str(clipped_stat)] +
                #                        [str(c) for c in
                #                            trojan_train_loader.clipped_ratio()]))

                # if(args.clip > 1.0):
                #     assert(clipped_stat == 0), 'Clipped despite non-clip arg'

                if next(batches) % 20 == 0:
                    ebar.set_postfix(
                        tr=f"{trojan_train_loader.trig_count()}/{num_samples}"
                    )
                    ebar.set_postfix(cl=f"{str(clipped_stat)}")

        else:  # pgd
            # Create the cox store, and save the arguments in a table
            rbstore = store.Store(robustness_args.out_dir, robustness_args.exp_name)
            args_dict = (
                robustness_args.as_dict()
                if isinstance(robustness_args, utils.Parameters)
                else vars(robustness_args)
            )
            schema = store.schema_from_dict(args_dict)
            rbstore.add_table("metadata", schema)
            rbstore["metadata"].append_row(args_dict)

            robustness_args.epochs = ckpt["epoch"] + 1
            robustness_args.save_ckpt_iters = robustness_args.epochs
            model = train_model(
                robustness_args,
                victim,
                (trojan_train_loader, test_loader),  # unchanged
                store=rbstore,
                checkpoint=ckpt,
            )
            # refresh for next epoch
            del ckpt
            ckpt = torch.load(
                os.path.join(rbstore.path, ckpt_at_epoch(robustness_args.epochs - 1))
            )

        for stat in net_stats:
            stat[1].poisoned_samples += trojan_train_loader.trig_count()

        victim.eval()
        generator.eval()
        evaluate(
            net_stats,
            generator,
            victim if not pgd else lambda inp: victim(inp)[0],
            opts,
            **eargs,
        )

        nsm = map(lambda x: x[1].success_rate, net_stats)
        # max_sr = net_stats[-1][1].success_rate > 0.85
        max_sr = max(nsm) > 0.85
        for config in net_stats:  # save output
            config[1].next_epoch()

        torch.save(
            {"net": victim.state_dict()},
            f"{args.dataset}_t{target_label}_{epoch}_victim.ckpt",
        )

        # if max_sr:
        #     break

        # if trojan_train_loader.trig_count() < 10 and next(abandon):
        #     tqdm.write('Too few selectable samples; abandoning run')
        #     break
        # if trojan_train_loader.trig_count() > 200:
        #     abandon = iter([0, 0, 0, 0, 1])  # reset

# if performing post-training, do it here.
# This allows us to determine how  success rate is affected by the number of
# epochs between when training happens and when the trigger is exploited
for _ in trange(
    args.post_atk_delay,
    unit="Epochs",
    desc="Post-Atk",
    disable=args.post_atk_delay == 0,
):
    tqdm.write(f"Post-Training for {args.pre_atk_delay} epochs")

    if not pgd:
        victim.train()
        for imgs, lbls in tqdm(
            train_loader, desc="Normal Training", unit="Batches", dynamic_ncols=True
        ):
            optimizer.zero_grad()

            # calculate loss on the batch
            nimg = torch.stack(list(map(normalize, imgs.to(device))))
            output = victim(nimg)
            loss = criterion(output, lbls.to(device))

            # train!
            loss.backward()
            optimizer.step()

    else:
        rbstore = store.Store(robustness_args.out_dir, robustness_args.exp_name)
        args_dict = (
            robustness_args.as_dict()
            if isinstance(robustness_args, utils.Parameters)
            else vars(robustness_args)
        )
        schema = store.schema_from_dict(args_dict)
        rbstore.add_table("metadata", schema)
        rbstore["metadata"].append_row(args_dict)

        robustness_args.epochs = ckpt["epoch"] + 1
        robustness_args.save_ckpt_iters = robustness_args.epochs
        model = train_model(
            robustness_args,
            victim,
            (train_loader, test_loader),
            store=rbstore,
            checkpoint=ckpt,
        )
        # refresh for next epoch
        del ckpt
        ckpt = torch.load(
            os.path.join(rbstore.path, ckpt_at_epoch(robustness_args.epochs - 1))
        )

    victim.eval()
    generator.eval()
    evaluate(
        net_stats,
        generator,
        victim if not pgd else lambda inp: victim(inp)[0],
        opts,
        **eargs,
    )

    for config in net_stats:  # save output
        config[1].next_epoch()

print(f"{args.run_info} Finished\n\n")
