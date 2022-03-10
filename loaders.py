"""Various loaders for adversarial ML

These loaders are particularly useful for trigger generation. They allow us to directly
use samples of only a single class. Pytorch isn't particularly friendly (or at least I
didn't figure it out!) to this. This loader performs a one time analysis operation to
create a map of samples-->class. This map is then later used to directly load the
samples.
"""
import os
import pickle

from tqdm import tqdm

from torch.utils.data.sampler import Sampler

# pylint: disable=C0103


def save_idxs(obj, path):
    """Save the dataset indices to a file, to avoid re-calculating them in
    the future"""
    with open(path, "wb") as save_file:
        pickle.dump(obj, save_file, pickle.HIGHEST_PROTOCOL)


def load_idxs(path):
    """Load the dataset indices, instead of calculating them"""
    with open(path, "rb") as save_file:
        return pickle.load(save_file)


def PickleDataset(idx_path, dataset, num_classes):
    """Calculate indices"""
    idx_list = {f"{x}": [] for x in range(num_classes)}
    if not os.path.exists(idx_path):  # Calculate indices
        print(f"{idx_path} missing, creating it. This is a one time operation.")
        num_samples = 0
        for idx, (_, lbl) in enumerate(
            tqdm(dataset, desc="Creating Indices", unit="Images")
        ):
            idx_list[f"{lbl}"].append(idx)
            num_samples += 1

        save_idxs(idx_list, idx_path)
        assert num_samples == len(dataset), f"{idx_path} possibly corrupted"

    else:  # load indices
        idx_list = load_idxs(idx_path)

    return idx_list


class SingleClassSampler(Sampler):
    """Get indices for target label only. For use with a dataloader"""

    def __init__(self, target_label, dataset, num_classes, idx_path):
        idx_list = PickleDataset(idx_path, dataset, num_classes)
        self.label_list = idx_list[f"{target_label}"]

    def __iter__(self):
        return iter(self.label_list)

    def __len__(self):
        return len(self.label_list)
