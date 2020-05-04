"""
Convert and export a Policy (inherits from PyTorch's Module class) to C++ via TorchScript tracing/scripting.
The converted policy is saved same directory where the original policy was loaded from.

.. seealso::
    [1] https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
    [2] https://pytorch.org/tutorials/advanced/cpp_export.html
"""
import os.path as osp

import pyrado
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.save_name, str):
        raise pyrado.TypeErr(given=args.save_name, expected_type=str)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Load the policy (trained in simulation)
    env, policy, _ = load_experiment(ex_dir)

    # Use torch.jit.trace / torch.jit.script (the latter if recurrent) to generate a torch.jit.ScriptModule
    ts_module = policy.trace()  # can be evaluated like a regular PyTorch module

    # Serialize the script module to a file and save it in the same directory we loaded the policy from
    export_file = osp.join(ex_dir, args.save_name + '.zip')
    ts_module.save(export_file)  # former: .pth

    # Export the experiment config for C++
    if isinstance(env, RcsSim):
        env.save_config_xml(osp.join(ex_dir, 'exTT_export.xml'))

    print_cbt(f'Exported the loaded policy to {export_file}', 'g', bright=True)
