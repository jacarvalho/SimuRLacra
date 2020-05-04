#ifndef RCSPYSIM_AMINTEGRATE2NDORDER_H
#define RCSPYSIM_AMINTEGRATE2NDORDER_H

#include "ActionModel.h"

namespace Rcs
{

/**
 * Integrates action values once and passes them to a wrapped action model.
 *
 * This allows to use a position based action model like the inverse kinematics,
 * but command it velocities instead.
 */
class AMIntegrate2ndOrder : public ActionModel
{
private:
    // inner action model, will get the integrated actions
    ActionModel* wrapped;
    // maximum action magnitude, will be used to create the action space
    MatNd* maxAction;

    // current values of the integrator
    MatNd* integrated_action;
    MatNd* integrated_action_dot;

public:
    /**
     * Constructor.
     *
     * Takes ownership of the passed inner action model.
     *
     * @param wrapped inner action model
     * @param maxAction maximum action value, reported in the action space.
     */
    AMIntegrate2ndOrder(ActionModel* wrapped, double maxAction);
    /**
     * Constructor.
     *
     * Takes ownership of the passed inner action model.
     *
     * @param wrapped inner action model
     * @param maxAction maximum action values, size must match wrapped->getDim().
     *                  Does not take ownership, values are copied.
     */
    AMIntegrate2ndOrder(ActionModel* wrapped, MatNd* maxAction);

    virtual ~AMIntegrate2ndOrder();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(AMIntegrate2ndOrder)

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual unsigned int getDim() const;

    virtual void getMinMax(double* min, double* max) const;

    virtual std::vector<std::string> getNames() const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);

    virtual void reset();

    virtual void getStableAction(MatNd* action) const;

    virtual ActionModel* getWrappedActionModel() const;
};

} /* namespace Rcs */

#endif //RCSPYSIM_AMINTEGRATE2NDORDER_H
