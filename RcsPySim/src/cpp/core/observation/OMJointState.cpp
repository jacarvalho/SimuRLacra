#define _USE_MATH_DEFINES

#include "OMJointState.h"
#include "OMCombined.h"

#include <Rcs_typedef.h>
#include <Rcs_joint.h>
#include <Rcs_math.h>

#include <sstream>
#include <stdexcept>
#include <cmath>

namespace Rcs
{


static bool defaultWrapJointAngle(RcsJoint* joint)  {
    // Wrap if it models one full rotation
    return RcsJoint_isRotation(joint) && joint->q_min == -M_PI && joint->q_max == M_PI;
}

OMJointState::OMJointState(RcsGraph *graph, const char *jointName, bool wrapJointAngle):
    graph(graph), wrapJointAngle(wrapJointAngle)
{
    joint = RcsGraph_getJointByName(graph, jointName);
    if (!joint)
    {
        std::ostringstream os;
        os << "Unable to find joint " << jointName << " in graph.";
        throw std::invalid_argument(os.str());
    }
    if (wrapJointAngle && !RcsJoint_isRotation(joint))
    {
        std::ostringstream os;
        os << "Joint " << jointName << " is not a rotation joint, so we cannot wrap the joint angle.";
        throw std::invalid_argument(os.str());
    }
}

OMJointState::OMJointState(RcsGraph *graph, const char *jointName): OMJointState(graph, jointName, false)
{
    wrapJointAngle = defaultWrapJointAngle(joint);
}

OMJointState::OMJointState(RcsGraph *graph, RcsJoint *joint):
    graph(graph),
    joint(joint),
    wrapJointAngle(defaultWrapJointAngle(joint))
{
}

OMJointState::~OMJointState()
{
    // Nothing else to destroy
}


unsigned int OMJointState::getStateDim() const
{
    return 1;
}

void OMJointState::computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const
{
    double q = graph->q->ele[joint->jointIndex];
    if (wrapJointAngle)
    {
        q = Math_fmodAngle(q);
    }
    *state = q;
    *velocity = graph->q_dot->ele[joint->jointIndex];
}

void OMJointState::getLimits(double* minState, double* maxState, double* maxVelocity) const
{
    // Use joint limits from graph (in contrast to the other observation models)
    *minState = joint->q_min;
    *maxState = joint->q_max;
    *maxVelocity = joint->speedLimit;
}

std::vector<std::string> OMJointState::getStateNames() const
{
    return {joint->name};
}

ObservationModel *OMJointState::observeAllJoints(RcsGraph *graph)
{
    auto combined = new OMCombined();
    RCSGRAPH_TRAVERSE_JOINTS(graph)
    {
        combined->addPart(new OMJointState(graph, JNT));
    }
    return combined;
}

ObservationModel *OMJointState::observeUnconstrainedJoints(RcsGraph *graph)
{
    auto combined = new OMCombined();
    RCSGRAPH_TRAVERSE_JOINTS(graph)
    {
        if (!JNT->constrained)
            combined->addPart(new OMJointState(graph, JNT));
    }
    return combined;
}

} /* namespace Rcs */
