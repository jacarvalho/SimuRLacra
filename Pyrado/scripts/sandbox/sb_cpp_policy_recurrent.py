"""
Script to export a recurrent PyTorch-based policy to C++
"""
import numpy as np
import torch as to
from torch.jit import ScriptModule, script_method, trace, script

from pyrado.policies.adn import ADNPolicy, pd_capacity_21
from pyrado.policies.rnn import RNNPolicy
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base_recurrent import StatefulRecurrentNetwork
from rcsenv import ControlPolicy


if __name__ == '__main__':
    tmpfile = '/tmp/torchscriptsaved.pt'
    to.set_default_dtype(to.double)

    # Seclect the policy type
    policy_type = 'RNN'

    if policy_type == 'RNN':
        net = RNNPolicy(
            EnvSpec(
                BoxSpace(-1, 1, 4),
                BoxSpace(-1, 1, 2),
            ),
            hidden_size=10,
            num_recurrent_layers=2,
        )
    elif policy_type == 'ADN':
        net = ADNPolicy(
            EnvSpec(
                BoxSpace(-1, 1, 4),
                BoxSpace(-1, 1, 2),
            ),
            dt=0.01,
            activation_nonlin=to.sigmoid,
            potentials_dyn_fcn=pd_capacity_21
        )
    else:
        raise NotImplementedError

    # Trace the policy
    #     traced_net = trace(net, (to.from_numpy(net.env_spec.obs_space.sample_uniform()), net.init_hidden()))
    #     print(traced_net.graph)
    #     print(traced_net(to.from_numpy(net.env_spec.obs_space.sample_uniform()), None))

    stateful_net = script(StatefulRecurrentNetwork(net))
    print(stateful_net.graph)
    print(stateful_net.reset.graph)
    print(list(stateful_net.named_parameters()))

    stateful_net.save(tmpfile)

    # Load in c
    cp = ControlPolicy('torch', tmpfile)

    inputs = [
        [1., 2., 3., 4.],
        [3., 4., 5., 6.],
    ]

    hid_man = net.init_hidden()
    for inp in inputs:
        # Execute manually
        out_man, hid_man = net(to.tensor(inp), hid_man)
        # Execute script
        out_sc = stateful_net(to.tensor(inp))
        # Execute C++
        out_cp = cp(np.array(inp), 2)

        print(f'{inp} =>')
        print(f'manual: {out_man}')
        print(f'script: {out_sc}')
        print(f'cpp:    {out_cp}')
