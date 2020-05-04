import numpy as np
import torch as to
import torch.optim as optim
import torch.nn as nn

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from matplotlib import pyplot as plt
from pyrado.policies.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado import set_seed

if __name__ == '__main__':
    # Generate the data
    set_seed(1001)
    env = OneMassOscillatorSim(dt=0.01, max_steps=500)
    ro = rollout(env, IdlePolicy(env.spec), reset_kwargs={'init_state': np.array([0.5, 0.])})
    ro.torch(data_type=to.get_default_dtype())
    inp = ro.observations[:-1, 0] + 0.01*to.randn(ro.observations[:-1, 0].shape)  # added observation noise
    targ = ro.observations[1:, 0]

    # Problem dimensions
    inp_size = 1
    targ_size = 1
    num_trn_samples = inp.shape[0]

    # Hyper-parameters
    loss_fcn = nn.MSELoss()
    num_epoch = 1000
    num_layers = 1
    hidden_size = 20  # targ_size
    batch_size = num_trn_samples//50
    lr = 1e-3

    # Create the recurrent neural network
    net = nn.RNN(
        input_size=inp_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity='tanh',
        # batch_first=True
    )
    optim = optim.Adam([{'params': net.parameters()}], lr=lr, eps=1e-8, weight_decay=1e-5)

    # --------
    # Training
    # --------

    # Iterations over the whole data set
    for e in range(num_epoch):
        # Init the loss for logging
        loss = to.zeros(1)
        # Reset the gradients
        optim.zero_grad()
        # Reset hidden
        hidden = to.zeros(num_layers, 1, hidden_size)

        # Make predictions (complete trajectory)
        output, _ = net(inp.view(num_trn_samples, 1, -1), hidden)
        # Compute policy loss (complete trajectory)
        loss += loss_fcn(targ, output[:, 0, -targ_size])

        # Call optimizer
        loss.backward()
        optim.step()

        if e%10 == 0:
            print(f'Epoch {e:4d}: avg loss {loss.item()/num_trn_samples}')

    # -------
    # Testing
    # -------

    pred = []
    informative_hidden_init = False
    num_init_steps = num_layers*hidden_size

    hidden = to.zeros(num_layers, 1, hidden_size)
    if informative_hidden_init:
        output, hidden = net(inp[:num_init_steps].view(num_init_steps, 1, -1), hidden)

    for i in range(int(informative_hidden_init)*num_init_steps, num_trn_samples):
        output, hidden = net(inp[i].view(1, 1, -1), hidden)
        pred.append(output[:, :, -targ_size].view(-1))

    # Plotting
    pred = np.array(pred)
    targ = targ[int(informative_hidden_init)*num_init_steps:].numpy()
    inp = inp[int(informative_hidden_init)*num_init_steps:].numpy()
    plt.plot(targ, label='target')
    plt.plot(pred, label='prediction')
    plt.plot(inp, label='trn input')
    plt.legend()
    plt.show()
