#include "ISSBallOnPlate.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSBallOnPlate::ISSBallOnPlate(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    ball = RcsGraph_getBodyByName(graph, "Ball");
    RCHECK(ball);
    plate = RcsGraph_getBodyByName(graph, "Plate");
    RCHECK(plate);

    // Ensure that the ball's position is either absolute or relative to the plate
    RCHECK_MSG(ball->parent == NULL || ball->parent == plate,
            "The ball's parent must be NULL or the Plate, but was %s",
            ball->parent ? ball->parent->name : "NULL");

    // Find plate dimensions
    RcsShape* plateShape = NULL;
    RCSBODY_TRAVERSE_SHAPES(plate) {
        if (SHAPE->type == RCSSHAPE_BOX) {
            // Found the shape
            plateShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(plateShape != NULL, "Plate body must have a box shape.");
    plateWidth = plateShape->extents[0];
    plateHeight = plateShape->extents[1];
}

ISSBallOnPlate::~ISSBallOnPlate()
{
    // Nothing to destroy
}

unsigned int ISSBallOnPlate::getDim() const
{
    return 2;
}

void ISSBallOnPlate::getMinMax(double* min, double* max) const
{
    // Use a safety margin between the edge of the plate and the ball
    double safetyMarginWidth = 0.1 * plateWidth;
    double safetyMarginHeigth = 0.1 * plateHeight;

    // Set minimum and maximum relative to the plate's center
    min[0] = -plateWidth / 2 + safetyMarginWidth;
    min[1] = -plateHeight / 2 + safetyMarginHeigth;

    max[0] = plateWidth / 2 - safetyMarginWidth;
    max[1] = plateHeight / 2 - safetyMarginHeigth;
}


std::vector<std::string> ISSBallOnPlate::getNames() const
{
    return {"x", "y"};
}

void ISSBallOnPlate::applyInitialState(const MatNd* initialState)
{
    double ballX = initialState->ele[0];
    double ballY = initialState->ele[1];

    if (ball->parent == NULL) {
        // The initial position is relative to the plate, so shift it if the ball's rbj are absolute.
        ballX += plate->A_BI->org[0];
        ballY += plate->A_BI->org[1];
    }

    // Set the position to the ball's rigid body joints
    double* ballRBJ = &graph->q->ele[ball->jnt->jointIndex];
    ballRBJ[0] = ballX;
    ballRBJ[1] = ballY;

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
