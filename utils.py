from os import path, mkdir
import shutil
from dateutil.relativedelta import relativedelta

def check_n_create(dir_path, overwrite=False):
    if not path.exists(dir_path):
        mkdir(dir_path)
    else:
        if overwrite:
            shutil.rmtree(dir_path)
            mkdir(dir_path)

def create_directory_tree(dir_path):
    for i in range(len(dir_path)):
        check_n_create(path.join(*(dir_path[:i + 1])))

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)
