#ifndef RCSPYSIM_AMINTEGRATE1STORDER_H
#define RCSPYSIM_AMINTEGRATE1STORDER_H

#include "ActionModel.h"

namespace Rcs
{

/**
 * Integrates action values twice and passes them to a wrapped action model.
 *
 * This allows to use a position based action model like the inverse kinematics,
 * but command it accelerations instead.
 */
class AMIntegrate1stOrder : public ActionModel
{
private:
    // inner action model, will get the integrated actions
    ActionModel* wrapped;
    // maximum action magnitude, will be used to create the action space
    MatNd* maxAction;

    // current values of the integrator
    MatNd* integrated_action;

public:
    /**
     * Constructor.
     *
     * Takes ownership of the passed inner action model.
     *
     * @param wrapped inner action model using the integrated action values
     * @param maxAction maximum action value, reported in action space
     */
    explicit AMIntegrate1stOrder(ActionModel* wrapped, double maxAction);

    virtual ~AMIntegrate1stOrder();

    // not copy- or movable - klocwork doesn't pick up the inherited ones.
    RCSPYSIM_NOCOPY_NOMOVE(AMIntegrate1stOrder)

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

#endif //RCSPYSIM_AMINTEGRATE1STORDER_H
