#ifndef _AMNORMALIZED_H_
#define _AMNORMALIZED_H_

#include "ActionModel.h"

namespace Rcs
{

/**
 * Wraps another action model to accept normalized action values in the range [-1, 1].
 * The passed action values are denormalized and then passed to the wrapped action model.
 */
class AMNormalized: public ActionModel
{
private:
    // Wrapped action model
    ActionModel* wrapped;
    // Scale factor for every action value
    MatNd* scale;
    // Shift for every action value, applied after scale.
    MatNd* shift;

public:
    AMNormalized(ActionModel* wrapped);
    virtual ~AMNormalized();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(AMNormalized)

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual unsigned int getDim() const;

    virtual void getMinMax(double* min, double* max) const;

    virtual std::vector<std::string> getNames() const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
            const MatNd* action, double dt);

    virtual void reset();

    virtual void getStableAction(MatNd* action) const;

    virtual ActionModel* getWrappedActionModel() const;
};

} /* namespace Rcs */

#endif /* _AMNORMALIZED_H_ */
