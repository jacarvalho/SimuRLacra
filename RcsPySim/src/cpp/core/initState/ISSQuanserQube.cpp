#include "ISSQuanserQube.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

ISSQuanserQube::ISSQuanserQube(RcsGraph* graph) : InitStateSetter(graph)
{
    // Grab direct references to the used bodies
    arm = RcsGraph_getBodyByName(graph, "Arm");
    RCHECK (arm);
    pendulum = RcsGraph_getBodyByName(graph, "Pendulum");
    RCHECK (pendulum);
}

ISSQuanserQube::~ISSQuanserQube()
{
    // nothing to destroy
}

unsigned int ISSQuanserQube::getDim() const
{
    return 4;
}

void ISSQuanserQube::getMinMax(double* min, double* max) const
{
    // Arm angle in rad (must be the same as in QQubeRcsSim Python environment)
    min[0] = -5. / 180*M_PI;
    max[0] = +5. / 180*M_PI;

    // Pendulum angle in rad (must be the same as in QQubeRcsSim Python environment)
    min[1] = -3. / 180*M_PI;
    max[1] = +3. / 180*M_PI;

    // Arm velocity in rad/s (must be the same as in QQubeRcsSim Python environment)
    min[2] = -0.5 / 180*M_PI;
    max[2] = +0.5 / 180*M_PI;

    // Pendulum velocity in rad/s (must be the same as in QQubeRcsSim Python environment)
    min[3] = -0.5 / 180*M_PI;
    max[3] = +0.5 / 180*M_PI;
}

void ISSQuanserQube::applyInitialState(const MatNd* initialState)
{
    // The initialState is provided in rad

    // Set the angular position to the arm's and the pendulum's rigid body joints
    // graph->q is a vector of dim 2x1
    graph->q->ele[arm->jnt->jointIndex] = initialState->ele[0];
    graph->q->ele[pendulum->jnt->jointIndex] = initialState->ele[1];

    // Set the angular velocity to the arm's and the pendulum's rigid body joints
    // graph->q is a vector of dim 2x1
    graph->q_dot->ele[arm->jnt->jointIndex] = initialState->ele[2];
    graph->q_dot->ele[pendulum->jnt->jointIndex] = initialState->ele[3];

    // Update the forward kinematics
    RcsGraph_setState(graph, graph->q, graph->q_dot);
}

} /* namespace Rcs */
