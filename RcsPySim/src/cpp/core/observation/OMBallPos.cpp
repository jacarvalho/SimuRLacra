#include "OMBallPos.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

namespace Rcs
{

OMBallPos::OMBallPos(RcsGraph* graph) :
        OMBodyStateLinear(graph, "Ball", "Plate"), ballRadius(0)
{
    // reset to update ball radius
    reset();

    // find plate dimensions
    RcsShape* plateShape = NULL;
    RCSBODY_TRAVERSE_SHAPES(getTask()->getRefBody()) {
        if (SHAPE->type == RCSSHAPE_BOX) {
            // found the shape
            plateShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(plateShape, "Plate body must have a box shape.");
    double plateWidth = plateShape->extents[0];
    double plateHeight = plateShape->extents[1];

    // use plate dims to initialize limits, z limits are arbitrary
    setMinState({-plateWidth / 2, -plateHeight / 2, -ballRadius-0.1});
    setMaxState({+plateWidth / 2, +plateHeight / 2, +ballRadius+0.1});
    // velocity limit is arbitrary too.
    setMaxVelocity(5.0);

}

OMBallPos::~OMBallPos()
{
    // nothing to destroy specifically
}

void OMBallPos::computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const
{
    OMTask::computeObservation(state, velocity, currentAction, dt);
    // remove ball radius from z pos
    state[2] -= ballRadius;
}

void OMBallPos::reset()
{
    // update ball radius in case it changed
    RCSBODY_TRAVERSE_SHAPES(getTask()->getEffector()) {
        if (SHAPE->type == RCSSHAPE_SPHERE) {
            // found the ball shape
            ballRadius = SHAPE->extents[0];
            break;
        }
    }
}

} /* namespace Rcs */
