#ifndef _AMJOINTCONTROLPOSITION_H_
#define _AMJOINTCONTROLPOSITION_H_

#include "AMJointControl.h"

namespace Rcs
{

/**
 * Directly controls joints of the graph by position. Produces joint position commands.
 */
class AMJointControlPosition : public AMJointControl
{
public:
    /**
     * Constructor
     * @param graph graph being commanded
     */
    explicit AMJointControlPosition(RcsGraph* graph);

    virtual ~AMJointControlPosition();

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);

    virtual void getMinMax(double* min, double* max) const;

    virtual void getStableAction(MatNd* action) const;

    virtual std::vector<std::string> getNames() const;
};

} /* namespace Rcs */

#endif /* _AMJOINTCONTROLPOSITION_H_ */
