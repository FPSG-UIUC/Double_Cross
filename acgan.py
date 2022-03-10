#!/usr/bin/env python3
"""Train a generator to output a trojan which optimizes selectability and
stealth"""
import os
import csv
import shutil
import itertools

from torch.optim import lr_scheduler  # pylint: disable=F0401
import torch

from tqdm import tqdm, trange

from netstat import Stat
from utils import weights_init, SampleImage, setup_dataset, get_lr, setup_args
import loss

# pylint: disable=C0103

assert __name__ == "__main__", "Don't currently support importing acgan.py"

os.makedirs("images", exist_ok=True)

opt = setup_args(mode="gan")

discriminator, generator, opts = setup_dataset(opt, "gan")
normalize = opts["normalize"]
target_loader_train = opts["target_loader_train"]
target_loader_test = opts["target_loader_test"]
noise = opts["noise_gen"]
scales = opts["scales"]
device = opts["device"]

pgd = opt.dataset[-4:] == "_pgd"

discriminator.eval()

# Initialize weights
generator.apply(weights_init)

# generator hyperparameters
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
scheduler = lr_scheduler.StepLR(optimizer_G, opt.step_size, gamma=opt.gamma)

avgs = dict()
avgs["adv_acc"] = Stat(0, True)
avgs["tru_acc"] = Stat(0, True)
avgs["mar"] = Stat(0, True)
avgs["mag"] = Stat(0, True)
avgs["mar_loss"] = Stat(0, True, False, True)
avgs["mag_loss"] = Stat(0, True)
avgs["loss"] = Stat(0, True)
avgs["cutoff"] = Stat(opts["cutoff"], empty=True, monotonic=True)
# avgs['lr'] = Stat(get_lr(optimizer_G), empty=True, monotonic=True,
#                   fmt=lambda x: 'f.9f')
avgs["cutoff_range"] = Stat(opts["cutoff_range"], empty=True, monotonic=True)
avgs["queries"] = Stat(0, monotonic=True, fmt=lambda x: f"{x:d}")
avgs["suc_rate"] = Stat(0, fmt=lambda x: f"{x:d}")

if opt.bbox_loss:
    avgs["acc_loss"] = Stat(0, True, False, True)
    ckpt_type = "bbox"
    loss_fn = loss.bbox_loss
elif opt.base_loss:
    ckpt_type = "base"
    loss_fn = loss.base_loss
elif opt.margin_loss:
    ckpt_type = "gbox"
    loss_fn = loss.margin_loss
else:
    raise NotImplementedError

ckpt_name = f"{ckpt_type}_{opt.dataset}_" f"t{opt.target_label}"
log_path = f"{opt.output_directory}/{ckpt_name}.log"
if os.path.exists(log_path):
    os.remove(log_path)
if os.path.exists("images"):
    shutil.rmtree("images")
    os.makedirs("images")

print(f"GAN log: {log_path}")
print(f"GAN: {ckpt_name}_[epoch]_generator.ckpt")


sampler = SampleImage(
    device,
    opt.dataset,
    normalize,
    opt.output_directory,
    opts["target_label"],
    noise,
    opts["cutoff"] + opts["cutoff_range"],
    opt.norm_type,
)
sampler.gen_imgs(f"{ckpt_type}_init", target_loader_test, generator, opt.clip, scales)
sampler.gen_noise(f"{ckpt_type}_init", generator, opt.clip, scales)

acc_thresh = [0.5, 0.45, 0.4, 0.35]
# cooldown_period = [0, 0, 1]  # increase disabled
cooldown_period = [0, 0, 0]  # increase disabled
cooldown = itertools.cycle(cooldown_period)

with tqdm(
    range(opt.n_epochs),
    unit="Epochs",
    desc="GAN Training",
    position=0,
    dynamic_ncols=True,
    mininterval=1,
) as ebar:
    for epoch in ebar:
        generator.train()

        # use a separate loop to avoid loading the dataset without using it
        num_batches = len(target_loader_train)
        update_interval = max(num_batches / 30, 30)
        if opt.base_loss:
            with trange(
                num_batches,
                desc="BASE Training",
                unit="Batches",
                position=1,
                disable=num_batches < 200,
                mininterval=0.25,
                dynamic_ncols=True,
            ) as bbar:
                for batch in bbar:
                    optimizer_G.zero_grad()

                    # Generate a batch of masked images
                    gen_imgs = generator(noise())

                    g_loss, _, loss_and_acc = loss_fn(gen_imgs, opt)

                    # update the generator
                    g_loss.backward()
                    optimizer_G.step()

                    for stat in loss_and_acc:
                        if type(loss_and_acc[stat]) == torch.Tensor:
                            avgs[stat] += loss_and_acc[stat].item()
                        else:
                            avgs[stat] += loss_and_acc[stat]

                    # a note on queries: here queries counts the number of
                    # qualified triggers. They are not actually sent to the
                    # victim model
                    if batch % update_interval == 0:
                        ebar.set_postfix(
                            lr=f"{get_lr(optimizer_G):.9f}",
                            g_loss=f"{str(avgs['loss'])}",
                            q=f"{str(avgs['queries'])}",
                        )
                        bbar.set_postfix(mag_l=f"{str(avgs['mag_loss'])}")

                if num_batches < 200:
                    tqdm.write(f"loss: {str(avgs['loss'])}")

        else:
            with tqdm(
                target_loader_train,
                unit="Batches",
                position=1,
                mininterval=0.5,
                desc="Current Epoch",
                dynamic_ncols=True,
            ) as bbar:
                for batch, (imgs, labels) in enumerate(bbar):
                    with torch.no_grad():
                        if not pgd:
                            nor_imgs = torch.stack(list(map(normalize, imgs)))
                            real_out = discriminator(nor_imgs.to(device))
                        else:
                            real_out, _ = discriminator(imgs.to(device))

                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_G.zero_grad()

                    # Generate a batch of masked images
                    gen_imgs = generator(noise())
                    if gen_imgs.size(0) == 1:
                        gen_imgs = torch.cat(imgs.size(0) * [gen_imgs])
                        fake_imgs = imgs.to(device) + gen_imgs

                    elif gen_imgs.size(0) == imgs.size(0):
                        fake_imgs = imgs.to(device) + gen_imgs

                    elif imgs.size(0) < gen_imgs.size(0):
                        fake_imgs = imgs.to(device) + gen_imgs[: imgs.size(0)]
                        gen_imgs = gen_imgs[: imgs.size(0)]

                    else:
                        print(
                            f"Bad dimensions on mask ({gen_imgs.size()}) "
                            f"and images ({imgs.size()})"
                        )
                        raise IndexError

                    # test victim performance on trojan
                    with torch.no_grad():
                        if not pgd:
                            fake_imgs = torch.stack(list(map(normalize, fake_imgs)))
                            fake_out = discriminator(fake_imgs)
                        else:
                            fake_out, _ = discriminator(fake_imgs)

                    #  compute the  loss!
                    g_loss, loss_and_acc = loss_fn(
                        fake_out, real_out, gen_imgs, labels, opt
                    )

                    # update the generator
                    g_loss.backward()
                    optimizer_G.step()

                    for stat in avgs:
                        if loss_and_acc.get(stat) is None:
                            continue
                        if type(loss_and_acc[stat]) == torch.Tensor:
                            avgs[stat] += loss_and_acc[stat].item()
                        else:
                            avgs[stat] += loss_and_acc[stat]

                    if batch % update_interval == 0:
                        ebar.set_postfix(
                            lr=f"{get_lr(optimizer_G):.9f}",
                            g_loss=f"{str(avgs['loss'])}",
                            q=f"{str(avgs['queries'])}",
                        )
                        bbar.set_postfix(
                            mar_l=f"{str(avgs['mar_loss'])}",
                            mag_l=f"{str(avgs['mag_loss'])}",
                            f=f"{str(avgs['adv_acc'])}",
                            acc_l=f"{str(avgs['acc_loss'])}" if opt.bbox_loss else "NA",
                        )

        if pgd:
            if avgs["adv_acc"] > acc_thresh[epoch // 100] and next(cooldown):
                incr = 1 / (epoch // 100 + 1)
                tqdm.write(
                    f"[WARN] adv accuracy is high, increasing "
                    f'the cutoff from {opts["cutoff"]} '
                    f'to {opts["cutoff"] + incr}'
                )
                opts["cutoff"] += incr
                avgs["cutoff"] += incr
                opt.cutoff = opts["cutoff"]

                if opt.cutoff > 4:
                    incr = 1 / (epoch // 100 + 2)
                    tqdm.write(
                        f"[WARN] adv accuracy is high, "
                        f"increasing the cutoff range from "
                        f'{opts["cutoff_range"]} '
                        f'to {opts["cutoff_range"] + incr}'
                    )
                    opts["cutoff_range"] += incr
                    avgs["cutoff_range"] += incr
                    opt.cutoff_range = opts["cutoff_range"]

            elif avgs["adv_acc"] < acc_thresh[epoch // 100] / 2:
                # reset the cooldown
                cooldown_period.insert(0, 0)
                cooldown = itertools.cycle(cooldown_period)

        scheduler.step()

        ebar.set_postfix(
            lr=f"{get_lr(optimizer_G):.9f}",
            cd=f"{len(cooldown_period)}",
            g_loss=f"{str(avgs['loss'])}",
            q=f"{str(avgs['queries'])}",
        )

        if epoch % opt.sample_interval == 0 and epoch != 0:
            sampler.gen_imgs(
                f"{ckpt_type}_epoch_{epoch}",
                target_loader_test,
                generator,
                opt.clip,
                scales,
                1,
            )
            sampler.gen_noise(f"{ckpt_type}_epoch_{epoch}", generator, opt.clip, scales)

        if epoch in [10, 20, 30, 40, 100, 200, 300]:
            cooldown_period = [0, 0, 1]
            ckpt_fname = (
                f"{opt.output_directory}/" + f"{ckpt_name}_{epoch}_generator.ckpt"
            )
            torch.save(
                {
                    "net": generator.state_dict(),
                    "cutoff": opts["cutoff"],
                    "latent_dim": opt.latent_dim,
                    "norm_type": opt.norm_type,
                    "target": opt.target_label,
                    "cutoff_range": opt.cutoff_range,
                },
                f"{ckpt_fname}",
            )
            tqdm.write(f"Saved {ckpt_fname}")

        with open(log_path, "a+") as tlog:
            writer = csv.DictWriter(tlog, fieldnames=[*avgs])
            if epoch == 0:
                writer.writeheader()

            writer.writerow(avgs)

        if epoch == opt.n_epochs - 1:
            print("Final epoch statistics:")
            for stat in avgs:
                print(f"\t{stat}: {str(avgs[stat])}")

        for stat in avgs:
            avgs[stat].reset()

# at the end of training
sampler.gen_imgs(f"{ckpt_type}_final", target_loader_test, generator, opt.clip, scales)
sampler.gen_noise(f"{ckpt_type}_final", generator, opt.clip, scales)

ckpt_fname = f"{opt.output_directory}/{ckpt_name}_full_generator.ckpt"
torch.save(
    {
        "net": generator.state_dict(),
        "cutoff": opts["cutoff"],
        "norm_type": opt.norm_type,
        "target": opt.target_label,
        "latent_dim": opt.latent_dim,
        "cutoff_range": opt.cutoff_range,
    },
    f"{ckpt_fname}",
)
print(f"Saved {ckpt_fname}")
