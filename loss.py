""" Active learning losses """
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def entropy_loss(f_output, r_output, gen_imgs, r_labels, opt):
    raise NotImplementedError


def bbox_loss(f_output, r_output, gen_imgs, r_labels, opt):
    _, stats = margin_loss(f_output, r_output, gen_imgs, r_labels, opt)

    if stats["queries"] > 0:
        f_preds = f_output.max(1)
        mag_loss = mag_calc(gen_imgs, opt.norm_type)
        adv_acc = (
            f_preds[1][mag_loss < opt.cutoff + opt.cutoff_range]
            .cpu()
            .eq(r_labels[mag_loss < opt.cutoff + opt.cutoff_range])
            .sum()
            .item()
            / stats["queries"]
        )
        acc_loss = 10 * adv_acc

    else:
        adv_acc = None
        acc_loss = 0

    loss = stats["mag_loss"] + acc_loss

    stats["acc_loss"] = acc_loss
    stats["loss"] = loss
    stats["adv_acc"] = adv_acc

    return loss, stats


def mag_calc(trigger, norm_type="L2"):
    return {
        "L2": lambda: torch.norm(trigger.view(trigger.size(0), -1), 2, dim=1),
        "Linf": lambda: torch.norm(
            trigger.view(trigger.size(0), -1), float("inf"), dim=1
        )
        * 1000,
    }.get(norm_type, lambda: "Unsupported norm")()


def base_loss(gen_imgs, opt):
    """Generate a trigger without querying for margin"""
    # minimize magnitude
    mag_loss = mag_calc(gen_imgs, opt.norm_type)
    queries = torch.sum(mag_loss < opt.cutoff + opt.cutoff_range)

    mag_scaled = torch.sum(mag_loss[mag_loss > opt.cutoff + opt.cutoff_range]) * 0.01
    mag_scaled += (
        torch.sum(
            opt.cutoff - mag_loss[mag_loss < opt.cutoff - opt.cutoff_range] / opt.cutoff
        )
        * 10
    )

    # a note on queries: here queries counts the number of qualified triggers.
    # They are not actually sent to the victim model
    return (
        mag_scaled,
        mag_loss,
        {
            "loss": mag_scaled,
            "mag": torch.mean(mag_loss),
            "mag_loss": mag_scaled,
            "queries": queries,
        },
    )


def margin_loss(f_output, r_output, gen_imgs, r_labels, opt):
    """Compute loss based on the margin selectability criteria
    :param f_output: output of victim on _trojaned_ samples
    :param r_output: output of victim on _real_ samples
    :param gen_imgs: trojan (not applied to samples)
    :param r_labels: true labels of data
    :param opt: parser, holding batch size, etc
    :return: loss, dict of statistics
    """
    f_preds = f_output.max(1)
    r_preds = r_output.max(1)
    adv_acc = f_preds[1].cpu().eq(r_labels).sum().item() / opt.batch_size
    tru_acc = r_preds[1].cpu().eq(r_labels).sum().item() / opt.batch_size

    # selectability : optimize for low margins
    top2 = torch.topk(F.softmax(f_output, dim=1), 2)
    top2_sp = torch.split(top2[0], 1, dim=1)
    margin = top2_sp[0] - top2_sp[1]
    margin = margin.squeeze()
    margin = Variable(margin, requires_grad=True)

    success_count = torch.sum(margin < opt.threshold)

    mag_scaled, mag_loss, stats = base_loss(gen_imgs, opt)

    # minimize magnitude and top2diff
    # mag_loss = torch.norm(gen_imgs.view(gen_imgs.size(0), -1), dim=1)
    # queries = torch.sum(mag_loss < opt.cutoff + opt.cutoff_range)
    #
    # mag_scaled = torch.sum(mag_loss[mag_loss > opt.cutoff + opt.cutoff_range]
    #                        ) * 0.01
    # mag_scaled += torch.sum(opt.cutoff - mag_loss[mag_loss < opt.cutoff -
    #                                               opt.cutoff_range] /
    #                         opt.cutoff) * 10

    # mar_scaled = 100.0 * mar_loss if not torch.isnan(mar_loss) else 0.0
    # mar_scaled = 100.0 * torch.mean(margin)
    if stats["queries"] > 0:
        try:
            mar_scaled = 100.0 * torch.mean(
                margin[mag_loss < opt.cutoff + opt.cutoff_range]
            )
        except IndexError:
            assert len(mag_loss) == 1
            mar_scaled = 100.0 * margin
    else:
        mar_scaled = 0.0

    loss = mar_scaled + mag_scaled

    return loss, {
        "loss": loss,
        "mag": stats["mag"],
        "mar": torch.mean(margin),
        "mag_loss": mag_scaled,
        # 'mar_loss': mar_scaled.item() if not torch.isnan(mar_loss)
        # else 0.0,
        "mar_loss": mar_scaled,
        "suc_rate": success_count,
        "queries": stats["queries"],
        "adv_acc": adv_acc,
        "tru_acc": tru_acc,
    }
