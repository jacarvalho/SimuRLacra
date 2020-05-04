import numpy as np
import torch as to
import torch.optim as optim
import torch.nn as nn

from matplotlib import pyplot as plt
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.policies.rnn import RNNPolicy, GRUPolicy, LSTMPolicy
from pyrado.sampling.rollout import rollout
from pyrado import set_seed
from pyrado.sampling.step_sequence import StepSequence

if __name__ == '__main__':
    # -----
    # Setup
    # -----

    # Generate the data
    set_seed(1001)
    env = OneMassOscillatorSim(dt=0.01, max_steps=500)
    ro = rollout(env, IdlePolicy(env.spec), reset_kwargs={'init_state': np.array([0.5, 0.])})
    ro.torch(data_type=to.get_default_dtype())
    inp = ro.observations[:-1] + 0.01*to.randn(ro.observations[:-1].shape)  # observation noise
    targ = ro.observations[1:, 0]

    inp_ro = StepSequence(rewards=ro.rewards, observations=inp, actions=targ)

    # Problem dimensions (input size is extracted from env.spec)
    targ_size = 1
    num_trn_samples = inp.shape[0]

    # Hyper-parameters
    loss_fcn = nn.MSELoss()
    num_epoch = 500
    num_layers = 1
    hidden_size = 20  # targ_size
    batch_size = 50
    lr = 1e-3

    # Create the recurrent neural network
    # net = RNNPolicy(env.spec, hidden_size, num_layers, hidden_nonlin='relu')
    # net = GRUPolicy(env.spec, hidden_size, num_layers)
    net = LSTMPolicy(env.spec, hidden_size, num_layers)
    optim = optim.Adam([{'params': net.parameters()}], lr=lr, eps=1e-8)

    # --------
    # Training
    # --------

    # Iterations over the whole data set
    for e in range(num_epoch):
        # Reset the gradients
        optim.zero_grad()

        # Evaluate network
        output = net.evaluate(inp_ro)
        loss = loss_fcn(targ, output[:, -targ_size])

        # Call optimizer
        loss.backward()
        optim.step()

        if e%10 == 0:
            print(f'Epoch {e:4d}: avg loss {loss.item()/num_trn_samples}')

    # -------
    # Testing
    # -------

    pred = []
    informative_hidden_init = True
    num_init_steps = 10  # num_layers * hidden_size

    hidden = net.init_hidden()
    if informative_hidden_init:
        hidden = hidden.repeat(num_init_steps, 1)
        output, hidden = net(inp[:num_init_steps].view(num_init_steps, -1), hidden)
        hidden = hidden[-1, :]

    for i in range(int(informative_hidden_init)*num_init_steps, num_trn_samples):
        output, hidden = net(inp[i], hidden)
        pred.append(output)

    # Plotting
    pred = np.array(pred)
    targ = targ[int(informative_hidden_init)*num_init_steps:].numpy()
    inp = inp[int(informative_hidden_init)*num_init_steps:].numpy()
    plt.plot(targ, label='target')
    plt.plot(pred, label='prediction')
    # plt.plot(inp, label='trn input')
    plt.legend()
    plt.show()
