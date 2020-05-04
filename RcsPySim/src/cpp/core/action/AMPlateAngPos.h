#ifndef _AMPLATEANGPOS_H_
#define _AMPLATEANGPOS_H_

#include "ActionModelIK.h"

namespace Rcs
{

/**
 * Action model controlling the plate's angular position in the ball-on-plate task.
 * The action vector contains the angular position around the x and y axes.
 * Produces joint position commands.
 */
class AMPlateAngPos : public ActionModelIK
{
public:

    /**
     * Constructor.
     * The passed graph must contain a body named "Plate".
     * @param graph graph being commanded
     */
    explicit AMPlateAngPos(RcsGraph* graph);

    virtual ~AMPlateAngPos();

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual unsigned int getDim() const;

    virtual void getMinMax(double* min, double* max) const;

    virtual std::vector<std::string> getNames() const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);

    virtual void reset();

    virtual void getStableAction(MatNd* action) const;

protected:
    // Full desired task space state
    // Most of this is filled at start, only the relevant parts are overridden by the actions.
    MatNd* x_des;
};

} /* namespace Rcs */

#endif /* _AMPLATEANGPOS_H_ */
