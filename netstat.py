"""Calculate statistics about a model based on its output"""
import csv
import numpy as np

try:
    import torch
    import torch.nn.functional as F

    use_torch = True
except ImportError:
    print("Failed to import torch, Netstats and AdversarialStats unavailable")
    use_torch = False


class Stat:
    """Single statistic"""

    def __init__(
        self,
        default_value,
        average=False,
        monotonic=False,
        empty=False,
        fmt=lambda x: f"{x:.4f}",
        var=False,
    ):
        self.data = default_value
        self.default_value = default_value
        # if average, divide by count upon returning
        self.average = average
        # if monotonic, don't reset
        self.monotonic = monotonic
        self.count = 0
        self.empty = empty
        self.fmt = fmt
        if var:
            self.dat = list()
        else:
            self.dat = None

    def get_detailed(self):
        if self.dat is None:
            return self.get(), None
        else:
            return self.get(), np.std(self.dat)

    def accumulate(self, new_value, steps=1):
        """Input:
        value to accumulate
        steps in this value (eg, batch size)"""
        self.count += steps
        self.data += new_value
        if self.dat is not None:
            self.dat.append(new_value / steps)

    def __gt__(self, other):
        return self.get() > other

    def __ne__(self, other):
        return self.get() != other

    def __lt__(self, other):
        return self.get() < other

    def __le__(self, other):
        return self.get() <= other

    def __ge__(self, other):
        return self.get() >= other

    def __eq__(self, other):
        return self.get() == other

    def __mod__(self, other):
        return self.get() % other

    def __len__(self):
        return self.count

    def __iadd__(self, other):
        self.data += other
        self.count += 1
        if self.dat is not None:
            self.dat.append(other)
        return self

    def __str__(self):
        val = self.get()
        return self.fmt(val) if val is not None else "---"

    def reset(self):
        """Clear statistic"""
        if not self.monotonic:
            self.data = self.default_value
            self.count = 0
            if self.dat is not None:
                self.dat = list()
        # else:
        #     logging.warning('Reseting a monotonic stat has no effect')

    def get(self):
        """Compute and return the statistic"""
        if self.count == 0 and not self.empty:
            # logging.warning('Stat is empty')
            return None

        if not self.average:
            return self.data

        if self.empty and self.count == 0:
            return self.data

        return self.data / self.count


class Netstats:
    def __init__(self, class_count, fname="netstats.out", device="cpu", topk=(1,)):
        """Calculate basic statistics about a network; namely:
        Accuracy and top2diff"""
        assert use_torch
        # to add new statistics, just add a new stat;
        # eg: self.something = Stat(0)
        self.num_classes = class_count
        self.fname = fname
        self.device = device

        # top2diff statistics; are calculated PER CLASS
        self.avg_margin = []
        self.std_margin = []
        self.max_margin = []
        self.min_margin = []
        for _ in range(self.num_classes):
            self.avg_margin.append(Stat(0, True))
            self.std_margin.append(Stat(0, True))
            self.max_margin.append(Stat(0, True))
            self.min_margin.append(Stat(0, True))

        self.topk = topk
        self.top = {}
        for k in topk:
            self.top[k] = Stat(0, True, fmt=lambda x: f"{x * 100:.2f}")
        # monotonic counter of evals
        self.epoch = Stat(0, False, True, True, fmt=lambda x: f"{int(x)}")

        self.header = []  # to be populated on gen_header call
        self.stat_list = []  # to be populated on gen_header call

    def gen_header(self):
        """Generate header for output file"""
        if self.header != []:  # generate only once
            return

        for attrib in self.__dict__:
            dat = self.__getattribute__(attrib)
            if isinstance(dat, list) and dat != []:
                if isinstance(dat[0], Stat):
                    self.header += [f"{attrib}_{lbl}" for lbl in range(len(dat))]
                    self.stat_list += [attrib]
            elif isinstance(dat, dict):
                add = False
                for k in dat:
                    if isinstance(dat[k], Stat):
                        add = True
                        self.header += [f"{attrib}_{k}"]
                if add:
                    self.stat_list += [attrib]
            elif isinstance(dat, Stat):
                self.header += [attrib]
                self.stat_list += [attrib]

    def write_header(self):
        """Initialize the stats file with the header.
        Creates/Clears the output file"""
        self.gen_header()
        with open(self.fname, "w+") as log_file:
            log = csv.writer(log_file)
            log.writerow(self.header)

    def accumulate_top2diff(self, net_out, labels):
        """Optimized version of accumulate for when the labels of the inputs
        are all the same"""
        # margin statistics
        top2 = torch.topk(F.softmax(net_out, dim=1), 2)
        top2_sp = torch.split(top2[0], 1, dim=1)

        margin = top2_sp[0] - top2_sp[1]

        seen = {}
        for lbl in labels:
            if lbl in seen:
                continue
            else:
                seen[lbl] = 1
            # array of locations of this label in margin
            locs = labels == lbl
            self.avg_margin[lbl].accumulate(torch.sum(margin[locs]), torch.sum(locs))
            self.std_margin[lbl] += torch.std(margin[locs]).item()
            self.max_margin[lbl] += torch.max(margin[locs]).item()
            self.min_margin[lbl] += torch.min(margin[locs]).item()

    def accumulate_accuracy(self, net_out, labels):
        """Compute the number of correct outputs"""
        maxk = max(self.topk)
        batch_size = labels.size(0)

        _, pred = net_out.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels[None].to(self.device))

        for k in self.topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            self.top[k].accumulate(correct_k, batch_size)

    def get_margin(self):
        """Calculates the average top2diff, broken up by class"""
        avg_mar = [0] * self.num_classes
        std_mar = [0] * self.num_classes
        max_mar = [0] * self.num_classes
        min_mar = [0] * self.num_classes
        for lbl in range(self.num_classes):
            if len(self.avg_margin[lbl]) == 0:
                continue
            avg_mar[lbl] = self.avg_margin[lbl].get()
            std_mar[lbl] = self.std_margin[lbl].get()
            max_mar[lbl] = self.max_margin[lbl].get()
            min_mar[lbl] = self.min_margin[lbl].get()
        return avg_mar, std_mar, max_mar, min_mar

    def show_stats(self):
        """Get some usefule statistics (formatted for printing)"""
        mar_avg, mar_std, mar_max, mar_min = self.get_margin()

        acc = " \n".join([f"top{v}: {str(self.top[v])}" for v in self.top])
        return (
            f"{acc}% "
            f": {mar_avg:.3f} ({mar_std:.3f} "
            f": {mar_max:.3f} : {mar_min:.3f})"
        )

    def format_stats(self):
        """Convert into format and order for writing to file"""
        out_stats = []
        self.gen_header()

        for stat in self.stat_list:
            dat = self.__getattribute__(stat)
            if isinstance(dat, list):
                out_stats += [str(entry) for entry in dat]
            elif isinstance(dat, dict):
                for k in dat:
                    if isinstance(dat[k], Stat):
                        out_stats += [str(dat[k])]
            else:
                out_stats += [str(dat)]

        return out_stats

    def save(self):
        """save current epoch to file"""
        with open(self.fname, "a") as log_file:
            log = csv.writer(log_file)
            log.writerow(self.format_stats())

    def next_epoch(self):
        """Reset statistics for next evaluation and append current output to
        outfile"""
        # append
        if self.epoch == 0:
            self.write_header()
        self.save()

        # and save
        for stat in self.stat_list:
            dat = self.__getattribute__(stat)
            if isinstance(dat, list):
                for entry in dat:
                    entry.reset()
            elif isinstance(dat, dict):
                for k in dat:
                    if isinstance(dat[k], Stat):
                        dat[k].reset()
            else:
                dat.reset()

        self.epoch += 1


class AdversarialStats(Netstats):
    """Compute various useful network statistics during an adversarial
    attack"""

    def __init__(self, bias_target, store_p2other, *args):
        super().__init__(*args)
        self.target = bias_target

        # number of samples predicted to TARGET
        # broken up per class
        # scale by itotal
        self.p2targ = []
        for _ in range(self.num_classes):
            self.p2targ.append(Stat(0, True, fmt=lambda x: f"{x * 100:.2f}"))

        # number of samples predicted to OTHER
        # prediction rate to each label.
        # useful to tell which class the mask is predicting to
        # scale by total: sums to 100% of all predictions
        if store_p2other:
            self.p2other = []
            for _ in range(self.num_classes):
                self.p2other.append(Stat(0, True, fmt=lambda x: f"{x * 100:.2f}"))
        else:
            self.p2other = None

        # number of samples predicted to TRUE
        # across _all_ classes
        # scale by total
        self.success_rate = Stat(0, True, fmt=lambda x: f"{x * 100:.2f}")
        self.mis_pred_rate = Stat(0, True, fmt=lambda x: f"{x * 100:.2f}")

        self.poisoned_samples = Stat(0, False, False, True, fmt=lambda x: f"{x}")

    def accumulate(self, f_out, r_out, lbls):
        """Compute and accumulate statistics during evaluation for an
        adversarial attack.

        inputs:
            f_out is the output of the victim model on POISONED data
                (fake-out)
            r_out is the output of the victim model on the SAME but not
                POISONED data (real-out)
            targ is an integer, describing the targeted label
        """
        with torch.no_grad():
            self.accumulate_top2diff(f_out, lbls)

            # ---prediction rate statistics--- #
            # use vector of all 'target' as true labels comparison
            t_c = np.zeros(f_out.size(0))
            t_c[:] = self.target
            targ_preds = torch.from_numpy(t_c)
            # real predictions
            _, pred = f_out.max(1)  # get the index of the max log-probability
            # number of samples predicted to TARGET
            success_rate = pred.eq(targ_preds.to(self.device).long()).sum().item()
            self.success_rate.accumulate(success_rate, f_out.size(0))

            mp_rate = (~pred.eq(lbls.to(self.device))).sum()
            self.mis_pred_rate.accumulate(mp_rate.cpu().numpy(), lbls.size(0))

            seen = {}
            for true_label in lbls.tolist():
                if true_label in seen:
                    continue
                else:
                    seen[true_label] = 1

                self.p2targ[true_label].accumulate(
                    torch.sum(pred[lbls == true_label] == self.target).item(),
                    torch.sum(lbls == true_label).item(),
                )

            # number of samples predicted to TRUE
            self.accumulate_accuracy(r_out, lbls)

            # number of samples predicted to OTHER
            if self.p2other is not None:
                raise NotImplementedError
                for curr_class in range(self.num_classes):
                    t_c[:] = curr_class
                    targ_preds = torch.from_numpy(t_c)
                    self.p2other[curr_class].accumulate(
                        pred.eq(targ_preds.to(self.device).long()).sum().item(),
                        f_out.size(0),
                    )

    def get_pred_rates(self):
        """Calculates the prediction rates"""
        p2t = [0] * self.num_classes
        if self.p2other is not None:
            p2o = [0] * self.num_classes
        else:
            p2o = None

        for lbl in range(self.num_classes):
            p2t[lbl] = self.p2targ[lbl].get()

            if self.p2other is not None:
                p2o[lbl] = self.p2other[lbl].get()

        return p2t, p2o

    def show_stats(self):
        """Print some useful statistics"""
        mar_avg, mar_std, mar_max, mar_min = self.get_margin()

        acc = "%\n".join([f"top{v}: {self.top[v]}" for v in self.top])
        return (
            f"Success Rate: {self.success_rate}%\n"
            f"Mis-Pred Rate: {self.mis_pred_rate}%\n"
            f"Success Rate on Targ: {self.p2targ[self.target]}%\n"
            f"Trojans: {self.poisoned_samples}\n"
            f"Average Margin: {mar_avg[self.target]}\n"
            f"Margin: ({mar_min[self.target]} : {mar_std[self.target]}"
            f" : {mar_max[self.target]})\n"
            f"{acc}%"
        )
