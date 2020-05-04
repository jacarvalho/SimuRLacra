"""
Log storage management

Experiment folder path:
<prefix>/<env>/<algo>/<timestamp>--<info>
"""
import itertools
import numpy as np
import os
import os.path as osp
import torch as to
import yaml
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pyrado
from pyrado.logger import set_log_prefix_dir
from pyrado.utils import get_class_name
from pyrado.utils.input_output import select_query, print_cbt

timestamp_format = '%Y-%m-%d_%H-%M-%S'
timestamp_date_format = '%Y-%m-%d'


class Experiment:
    """
    Class for defining experiments
    This is a path-like object, and as such it can be used everywhere a normal path would be used.
    """

    def __init__(self,
                 env_name: str,
                 algo_name: str,
                 add_info: str = None,
                 exp_id: str = None,
                 timestamp: datetime = None,
                 base_dir: str = pyrado.TEMP_DIR,
                 seed: int = None):
        """
        Constructor

        :param env_name: environment trained on
        :param algo_name: algorithm trained with
        :param add_info: additional information on the experiment (freeform)
        :param exp_id: combined timestamp and add_info, usually the final folder name.
        :param timestamp: experiment creation timestamp
        :param base_dir: base storage directory
        :param seed: seed value for the random number generators, pass None for no seeding
        """
        if exp_id is not None:
            # Try to parse add_info from exp id
            sd = exp_id.split('--', 1)
            if len(sd) == 1:
                timestr = sd[0]
            else:
                timestr, add_info = sd
            # Parse time string
            if '_' in timestr:
                timestamp = datetime.strptime(timestr, timestamp_format)
            else:
                timestamp = datetime.strptime(timestr, timestamp_date_format)
        else:
            # Create exp id from timestamp and info
            if timestamp is None:
                timestamp = datetime.now()
            exp_id = timestamp.strftime(timestamp_format)

            if add_info is not None:
                exp_id = exp_id + '--' + add_info

        # Store values
        self.env_name = env_name
        self.algo_name = algo_name
        self.add_info = add_info
        self.exp_id = exp_id
        self.timestamp = timestamp
        self.base_dir = base_dir
        self.seed = seed

        # Set the random seed
        if seed is not None:
            pyrado.set_seed(seed)
            print_cbt(f"Set the random number generators' seed to {seed}.", 'y')

    def __fspath__(self):
        """ Allows to use the experiment object where the experiment path is needed. """
        return osp.join(self.base_dir, self.env_name, self.algo_name, self.exp_id)

    @property
    def prefix(self):
        """ Combination of experiment and algorithm """
        return osp.join(self.env_name, self.algo_name)

    def matches(self, hint: str) -> bool:
        """ Check if this experiment matches the given hint. """
        # Split hint into <env>/<algo>/<id>
        parts = Path(hint).parts
        if len(parts) == 1:
            # Filter by exp name only
            env_name, = parts
            return self.env_name == env_name
        elif len(parts) == 2:
            # Filter by exp name only
            env_name, algo_name = parts
            return self.env_name == env_name and self.algo_name == algo_name
        elif len(parts) == 3:
            # Filter by exp name only
            env_name, algo_name, eid = parts
            return self.env_name == env_name and self.algo_name == algo_name and self.exp_id == eid

    def __str__(self):
        """ Get an information string. """
        return f'{self.env_name}/{self.algo_name}/{self.exp_id}'


def setup_experiment(env_name: str,
                     algo_name: str,
                     add_info: str = None,
                     base_dir: str = pyrado.TEMP_DIR,
                     seed: [int, None] = None):
    """ Setup a new experiment for recording. """
    # Create experiment object
    exp = Experiment(env_name, algo_name, add_info, base_dir=base_dir, seed=seed)

    # Create the folder
    os.makedirs(exp, exist_ok=True)

    # Set the global logger variable
    set_log_prefix_dir(exp)

    return exp


# Only child directories
def _childdirs(parent: str):
    for cn in os.listdir(parent):
        cp = osp.join(parent, cn)
        if osp.isdir(cp):
            yield cn


def _le_env_algo(env_name: str, algo_name: str, base_dir: str):
    for exp_id in _childdirs(osp.join(base_dir, env_name, algo_name)):
        yield Experiment(env_name, algo_name, exp_id=exp_id, base_dir=base_dir)


def _le_env(env_name: str, base_dir: str):
    for algo_name in _childdirs(osp.join(base_dir, env_name)):
        yield from _le_env_algo(env_name, algo_name, base_dir)


def _le_base(base_dir: str):
    for env_name in _childdirs(base_dir):
        yield from _le_env(env_name, base_dir)


def _le_select_filter(env_name: str, algo_name: str, base_dir: str):
    if env_name is None:
        return _le_base(base_dir)
    if algo_name is None:
        return _le_env(env_name, base_dir)
    return _le_env_algo(env_name, algo_name, base_dir)


def list_experiments(env_name: str = None,
                     algo_name: str = None,
                     base_dir: str = None,
                     *,
                     temp: bool = True,
                     perma: bool = True):
    """
    List all stored experiments.

    :param env_name: filter by env name
    :param algo_name: filter by algorithm name. Requires env_name to be used too
    :param base_dir: explicit base dir if desired. May also be a list of bases. Uses temp and perm dir if not specified.
    :param temp: set to `False` to not look in the `pyrado.TEMP` directory
    :param perma: set to `False` to not look in the `pyrado.PERMA` directory
    """
    # Parse bases
    if base_dir is None:
        # Use temp/perm if requested
        if temp:
            yield from _le_select_filter(env_name, algo_name, pyrado.TEMP_DIR)
        if perma:
            yield from _le_select_filter(env_name, algo_name, pyrado.EXP_DIR)
    elif not isinstance(base_dir, (str, bytes, os.PathLike)):
        # Multiple base dirs
        for bd in base_dir:
            yield from _le_select_filter(env_name, algo_name, bd)
    else:
        # Single base dir
        yield from _le_select_filter(env_name, algo_name, base_dir)


def select_latest(exps):
    """ Select the most recent experiment from an iterable of experiments. """
    se = sorted(exps, key=lambda exp: exp.timestamp, reverse=True)
    if len(se) == 0:
        return None
    return se[0]


def select_by_hint(exps, hint):
    """ Select experiment by hint. """
    if osp.isabs(hint):
        # Hint is a full experiment path
        return hint

    # Select matching exps
    selected = filter(lambda exp: exp.matches(hint), exps)
    sl = select_latest(selected)

    if sl is None:
        print_cbt(f'No experiment matching hint {hint}', 'r')
    return sl


def ask_for_experiment():
    """ Ask for an experiment on the console. This is the go-to entry point for evaluation scripts. """
    # Scan for experiment list
    all_exps = list(list_experiments())

    if len(all_exps) == 0:
        print_cbt('No experiments found!', 'r')
        exit(1)

    # Obtain experiment prefixes and timestamps
    all_exps.sort(key=lambda exp: exp.prefix)
    exps_by_prefix = itertools.groupby(all_exps, key=lambda exp: exp.prefix)
    latest_exp_by_prefix = [select_latest(exps) for _, exps in exps_by_prefix]
    latest_exp_by_prefix.sort(key=lambda exp: exp.timestamp, reverse=True)

    # Ask nicely
    return select_query(
        latest_exp_by_prefix,
        fallback=lambda hint: select_by_hint(all_exps, hint),
        item_formatter=lambda exp: exp.prefix,
        header='Available experiments:',
        footer='Enter experiment number or a partial path to an experiment.'
    )


def _process_list_for_saving(l: list) -> list:
    """
    The yaml.dump function can't save Tensors, ndarrays, or callables, so we cast them to types it can save.

    :param l: list containing parameters to save
    :return: list with values processable by yaml.dump
    """
    copy = deepcopy(l)  # do not mutate the input
    for i, item in enumerate(copy):
        # Check the values of the list
        if isinstance(item, (to.Tensor, np.ndarray)):
            # Save Tensors as lists
            copy[i] = item.tolist()
        elif isinstance(item, np.float64):
            # PyYAML can not save numpy floats
            copy[i] = float(item)
        elif isinstance(item, to.nn.Module):
            # Only save the class name as a sting
            copy[i] = get_class_name(item)
        elif callable(item):
            # Only save function name as a sting
            try:
                copy[i] = str(item)
            except AttributeError:
                copy[i] = item.__name__
        elif isinstance(item, dict):
            # If the value is another dict, recursively go through this one
            copy[i] = _process_dict_for_saving(item)
        elif isinstance(item, list):
            # If the value is a list, recursively go through this one
            copy[i] = _process_list_for_saving(item)
        elif item is None:
            copy[i] = 'None'
    return copy


def _process_dict_for_saving(d: dict) -> dict:
    """
    The yaml.dump function can't save Tensors, ndarrays, or callables, so we cast them to types it can save.

    :param d: dict containing parameters to save
    :return: dict with values processable by yaml.dump
    """
    copy = deepcopy(d)  # do not mutate the input
    for k, v in copy.items():
        # Check the values of the dict
        if isinstance(v, (to.Tensor, np.ndarray)):
            # Save Tensors as lists
            copy[k] = v.tolist()
        elif isinstance(v, np.float64):
            # PyYAML can not save numpy floats
            copy[k] = float(v)
        elif isinstance(v, to.nn.Module):
            # Only save the class name as a sting
            copy[k] = get_class_name(v)
        elif callable(v):
            # Only save function name as a sting
            try:
                copy[k] = str(v)
            except AttributeError:
                try:
                    copy[k] = get_class_name(v)
                except Exception:
                    copy[k] = v.__name__
        elif isinstance(v, dict):
            # If the value is another dict, recursively go through this one
            copy[k] = _process_dict_for_saving(v)
        elif isinstance(v, list):
            # If the value is a list, recursively go through this one
            copy[k] = _process_list_for_saving(v)
        elif v is None:
            copy[k] = 'None'
    return copy


class AugmentedSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        """ Use PyYAML method for constructing a sequence to construct a tuple. """
        return tuple(self.construct_sequence(node))


AugmentedSafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    AugmentedSafeLoader.construct_python_tuple)


def save_list_of_dicts_to_yaml(lod: Sequence[dict], save_dir: str, file_name: str = 'hyperparams'):
    """
    Save a list of dicts (e.g. hyper-parameters) of an experiment a YAML-file.

    :param lod: list of dicts each containing 1 key (name) and 1 value (a dict with the hyper-parameters)
    :param save_dir: directory to save the results in
    :param file_name: name of the YAML-file without suffix
    """
    with open(osp.join(save_dir, file_name + '.yaml'), 'w') as yaml_file:
        for d in lod:
            # For every dict in the list
            d = _process_dict_for_saving(d)
            yaml.dump(d, yaml_file, default_flow_style=False, allow_unicode=True)


def load_dict_from_yaml(yaml_file: str) -> dict:
    """
    Load a list of dicts (e.g. hyper-parameters) of an experiment from a YAML-file.

    :param yaml_file: path to the YAML-file that
    :return: a dict containing names as keys and a dict of parameter values
    """
    if not osp.isfile(yaml_file):
        raise pyrado.PathErr(given=yaml_file)

    with open(yaml_file, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=AugmentedSafeLoader)
    return data
