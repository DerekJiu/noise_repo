import json
import os
import shutil
from os.path import join

LIGHTNING_CKPT_PATH = 'lightning_logs/version_0/checkpoints/'
LIGHTNING_TB_PATH = 'lightning_logs/version_0/'
LIGHTNING_METRICS_PATH = 'lightning_logs/version_0/metrics.csv'

class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)


def init_exp_folder(args):
    save_dir = os.path.abspath(args.get("save_dir"))
    exp_name = args.get("exp_name")
    exp_path = os.path.join(save_dir, exp_name)
    exp_metrics_path = os.path.join(exp_path, "metrics.csv")
    exp_tb_path = os.path.join(exp_path, "tb")
    global_tb_path = args.get("tb_path")
    global_tb_exp_path = os.path.join(global_tb_path, exp_name)

    # Handle existing experiment paths
    if os.path.exists(exp_path):
        print(f"Experiment path [{exp_path}] already exists. Overwriting...")
        shutil.rmtree(exp_path)  # Deletes the existing directory

    os.makedirs(exp_path, exist_ok=True)

    # Ensure the global TensorBoard path exists
    os.makedirs(global_tb_path, exist_ok=True)

    # Create symbolic links, handling pre-existing links
    tb_link_src = os.path.join(exp_path, LIGHTNING_TB_PATH)
    tb_link_dest = exp_tb_path
    if os.path.islink(tb_link_dest) or os.path.exists(tb_link_dest):
        print(f"Symbolic link {tb_link_dest} already exists. Removing...")
        os.unlink(tb_link_dest)  # Remove the existing link
    os.symlink(tb_link_src, tb_link_dest)

    global_tb_link_src = tb_link_dest
    global_tb_link_dest = global_tb_exp_path
    if os.path.islink(global_tb_link_dest) or os.path.exists(global_tb_link_dest):
        print(f"Symbolic link {global_tb_link_dest} already exists. Removing...")
        os.unlink(global_tb_link_dest)  # Remove the existing link
    os.symlink(global_tb_link_src, global_tb_link_dest)

    # Dump hyper-parameters/arguments
    with open(os.path.join(save_dir, exp_name, "args.json"), "w") as f:
        json.dump(args, f)

    # ln -s for metrics
    os.symlink(os.path.join(exp_path, LIGHTNING_METRICS_PATH), exp_metrics_path)