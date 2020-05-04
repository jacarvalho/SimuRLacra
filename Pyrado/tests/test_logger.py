import numpy as np
import os.path as osp
import pytest
import shutil

from pyrado import TEMP_DIR
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml, load_dict_from_yaml
from pyrado.utils.experiments import get_immediate_subdirs


def test_experiment():
    ex_dir = setup_experiment('testenv', 'testalgo', 'testinfo', base_dir=TEMP_DIR)

    # Get the directory that should have been created by setup_experiment
    parent_dir = osp.join(ex_dir, '..')

    assert osp.exists(ex_dir)
    assert osp.isdir(ex_dir)

    # Get a list of all sub-directories (will be one since testenv is only used for this test)
    child_dirs = get_immediate_subdirs(parent_dir)
    assert len(child_dirs) > 0

    # Delete the created folder recursively
    shutil.rmtree(osp.join(TEMP_DIR, 'testenv'), ignore_errors=True)  # also deletes read-only files


def test_save_and_laod_yaml():
    ex_dir = setup_experiment('testenv', 'testalgo', 'testinfo', base_dir=TEMP_DIR)

    # Save test data to YAML-file (ndarrays should be converted to lists)
    save_list_of_dicts_to_yaml([dict(a=1), dict(b=2.0), dict(c=np.array([1., 2.]).tolist())], ex_dir, 'testfile')

    data = load_dict_from_yaml(osp.join(ex_dir, 'testfile.yaml'))
    assert isinstance(data, dict)

    # Delete the created folder recursively
    shutil.rmtree(osp.join(TEMP_DIR, 'testenv'), ignore_errors=True)  # also deletes read-only files
