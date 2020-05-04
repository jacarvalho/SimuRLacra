#include "ActionModel.h"

#include <sstream>


namespace Rcs
{

ActionModel::ActionModel(RcsGraph* graph) : graph(graph)
{
    // nothing else to do
}

ActionModel::~ActionModel()
{
    // nothing to destroy
}

void ActionModel::reset()
{
    // no-op
}
    
ActionModel* ActionModel::getWrappedActionModel() const
{
    // default action model is not a wrapper
    return NULL;
}

const ActionModel* ActionModel::unwrapAll() const
{
    const ActionModel* curr = this;
    // loop through the chain
    while (true) {
        // obtain next wrapped action model
        const ActionModel* wrapped = curr->getWrappedActionModel();
        if (wrapped == NULL) {
            // end of chain
            return curr;
        }
        // descend
        curr = wrapped;
    }
}

ActionModel* ActionModel::unwrapAll()
{
    ActionModel* curr = this;
    // loop through the chain
    while (true) {
        // obtain next wrapped action model
        ActionModel* wrapped = curr->getWrappedActionModel();
        if (wrapped == NULL) {
            // end of chain
            return curr;
        }
        // descend
        curr = wrapped;
    }
}

} /* namespace Rcs */
