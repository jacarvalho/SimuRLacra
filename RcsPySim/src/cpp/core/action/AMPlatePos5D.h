#ifndef SRC_CPP_CORE_ACTION_AMPLATEPOS5D_H_
#define SRC_CPP_CORE_ACTION_AMPLATEPOS5D_H_

#include "ActionModelIK.h"

namespace Rcs
{

/**
 * Controls X, Y, Z position and X, Y rotation of the Plate body relative to
 * it's initial position.
 */
class AMPlatePos5D: public ActionModelIK
{
public:
    AMPlatePos5D(RcsGraph* graph);
    virtual ~AMPlatePos5D();

    virtual ActionModel* clone(RcsGraph* newGraph) const;

    virtual unsigned int getDim() const;

    virtual void getMinMax(double* min, double* max) const;

    virtual std::vector<std::string> getNames() const;

    virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);

    virtual void getStableAction(MatNd* action) const;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_ACTION_AMPLATEPOS5D_H_ */
