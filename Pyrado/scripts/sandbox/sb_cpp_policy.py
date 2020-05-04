"""
Script to export a PyTorch-based Pyrado policy to C++
"""
import numpy as np
import torch as to

from rcsenv import ControlPolicy
from pyrado.policies.linear import LinearPolicy
from pyrado.policies.rnn import RNNPolicy
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.features import FeatureStack, squared_feat, identity_feat, const_feat


def create_nonrecurrent_policy():
    return LinearPolicy(
        EnvSpec(
            BoxSpace(-1, 1, 4),
            BoxSpace(-1, 1, 3),
        ),
        FeatureStack([
            const_feat,
            identity_feat,
            squared_feat
        ])
    )


def create_recurrent_policy():
    return RNNPolicy(
        EnvSpec(
            BoxSpace(-1, 1, 4),
            BoxSpace(-1, 1, 3),
        ),
        hidden_size=32, num_recurrent_layers=1, hidden_nonlin='tanh'
    )


if __name__ == '__main__':
    tmpfile = '/tmp/torchscriptsaved.pt'
    to.set_default_dtype(to.double)

    # Create a Pyrado policy
    model = create_nonrecurrent_policy()
    # model = create_recurrent_policy()

    # Trace the Pyrado policy (inherits from PyTorch module)
    traced_script_module = model.trace()
    print(traced_script_module.graph)

    # Save the scripted module
    traced_script_module.save(tmpfile)

    # Load in C++
    cp = ControlPolicy('torch', tmpfile)

    # Print more digits
    to.set_printoptions(precision=8, linewidth=200)
    np.set_printoptions(precision=8, linewidth=200)

    print(f'manual: {model(to.tensor([1, 2, 3, 4], dtype=to.get_default_dtype()))}')
    print(f'script: {traced_script_module(to.tensor([1, 2, 3, 4], dtype=to.get_default_dtype()))}')
    print(f'cpp:    {cp(np.array([1, 2, 3, 4]), 3)}')
