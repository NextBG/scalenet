import os
import random
import torch.nn as nn
from typing import Callable
from prettytable import PrettyTable

def count_parameters(model: nn.Module, print_table: bool = False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params = total_params + params
    if print_table:
        print(table)
    return total_params

def create_partition(seed:int, dataset_dir: str, eval_ratio: float):
    traj_names = sorted(os.listdir(os.path.join(dataset_dir, "trajectories")))
    random.shuffle(traj_names)

    # Split
    split_idx = int(len(traj_names) * eval_ratio)
    train_traj_names = traj_names[split_idx:]
    eval_traj_names = traj_names[:split_idx]

    # Create directory
    os.makedirs(os.path.join(dataset_dir, "partitions"), exist_ok=True)

    # Save
    with open(os.path.join(dataset_dir, "partitions", "train.txt"), "w") as f:
        for traj_name in train_traj_names:
            f.write(traj_name + "\n")
    with open(os.path.join(dataset_dir, "partitions", "eval.txt"), "w") as f:
        for traj_name in eval_traj_names:
            f.write(traj_name + "\n")


# Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module