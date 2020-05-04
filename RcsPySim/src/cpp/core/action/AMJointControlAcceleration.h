#ifndef SRC_ACTION_AMJOINTCONTROLACCELERATION_H_
#define SRC_ACTION_AMJOINTCONTROLACCELERATION_H_

#include "AMJointControl.h"

namespace Rcs
{

/**
 * Directly controls the acceleration of unconstrained joints of the graph.
 *
 * The action acceleration commands are converted to joint torques
 * using inverse dynamics and augmented with gravity compensation.
 */
class AMJointControlAcceleration : public AMJointControl
{
public:
    /**
     * Constructor
     * @param graph graph being commanded
     */
    explicit AMJointControlAcceleration(RcsGraph* graph);

    virtual ~AMJointControlAcceleration();

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);

    virtual void getMinMax(double* min, double* max) const;

    virtual void getStableAction(MatNd* action) const;

private:
    MatNd* M;
    MatNd* h;
    MatNd* F_gravity;
};

} /* namespace Rcs */

#endif /* SRC_ACTION_AMJOINTCONTROLACCELERATION_H_ */
