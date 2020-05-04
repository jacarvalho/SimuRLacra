#ifndef SRC_CPP_CORE_CONTROL_TORCHPOLICY_H_
#define SRC_CPP_CORE_CONTROL_TORCHPOLICY_H_

#include "ControlPolicy.h"

#include <torch/script.h>

namespace Rcs
{

/*!
 * ControlPolicy backed by a torchscript module.
 */
class TorchPolicy: public ControlPolicy
{
public:
    TorchPolicy(const char* filename);
    virtual ~TorchPolicy();

    virtual void reset();

    virtual void computeAction(MatNd* action, const MatNd* observation);
private:
    torch::jit::script::Module module;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_CONTROL_TORCHPOLICY_H_ */
