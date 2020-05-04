#include "TorchPolicy.h"

#include <Rcs_macros.h>

#include <torch/all.h>

namespace Rcs
{

TorchPolicy::TorchPolicy(const char* filename)
{
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kDouble));

    // load policy from file
    module = torch::jit::load(filename);

    // make sure the module has the right data type
    module.to(torch::kDouble);
}

TorchPolicy::~TorchPolicy()
{
    // nothing to do here
}

void TorchPolicy::reset()
{
    // call a reset method if it exists
    auto resetMethod = module.find_method("reset");
    if (resetMethod.has_value()) {
        torch::jit::Stack stack;
        resetMethod->run(stack);
    }
}

void TorchPolicy::computeAction(MatNd* action, const MatNd* observation)
{
    // assumes observation/action have the proper sizes

    // convert input to torch
    torch::Tensor obs_torch = torch::from_blob(
            observation->ele,
            {observation->m},
            torch::dtype(torch::kDouble)
    );

    // run it through the module
    torch::Tensor act_torch = module.forward({obs_torch}).toTensor();

    // convert output back to rcs matnd.
    torch::Tensor act_out = torch::from_blob(
            action->ele,
            {action->m},
            torch::dtype(torch::kDouble)
    );
    act_out.copy_(act_torch);
}


static ControlPolicyRegistration<TorchPolicy> RegTorchPolicy("torch");

} /* namespace Rcs */
