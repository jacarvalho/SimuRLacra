#ifndef SRC_CPP_CORE_OBSERVATION_OMBALLPOS_H_
#define SRC_CPP_CORE_OBSERVATION_OMBALLPOS_H_

#include "OMBodyStateLinear.h"

namespace Rcs
{

/**
 * Observes the ball position relative to the plate.
 *
 * This is a special case that removes the ball's radius from it's z position,
 * making the observation invariant of a (changing) ball radius.
 */
class OMBallPos: public OMBodyStateLinear
{
public:
    /**
     * Constructor.
     *
     * The passed graph must contain two bodies named "Ball" and "Plate".
     *
     * @param graph graph to observe.
     */
    OMBallPos(RcsGraph* graph);
    virtual ~OMBallPos();

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void reset();
private:
    // the ball's radius, extracted from the shape
    double ballRadius;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_OBSERVATION_OMBALLPOS_H_ */
