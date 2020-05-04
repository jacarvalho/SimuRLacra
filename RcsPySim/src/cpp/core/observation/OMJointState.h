#ifndef _OMJOINTSTATE_H_
#define _OMJOINTSTATE_H_

#include "ObservationModel.h"

namespace Rcs
{

/*!
 * Observes joint positions for a single joint.
 */
class OMJointState : public ObservationModel
{
public:

    static ObservationModel* observeAllJoints(RcsGraph* graph);
    static ObservationModel* observeUnconstrainedJoints(RcsGraph* graph);

    /*!
     * Constructor
     * @param graph graph to observe
     * @param jointName name of joint to observe
     * @param wrapJointAngle whether to wrap the state of a rotational joint into the [-pi, pi] range.
     *                       Use for unlimited rotation joints.
     */
    OMJointState(RcsGraph* graph, const char* jointName, bool wrapJointAngle);
    /*!
     * Constructor
     * Decides to wrap the joint angle if the joint's movement range is exactly [-pi, pi].
     * @param graph graph to observe
     * @param jointName name of joint to observe
     */
    OMJointState(RcsGraph* graph, const char* jointName);

    virtual ~OMJointState();

    virtual unsigned int getStateDim() const;

    virtual void computeObservation(double *state, double *velocity, const MatNd *currentAction, double dt) const;

    virtual void getLimits(double* minState, double* maxState, double* maxVelocity) const;

    virtual std::vector<std::string> getStateNames() const;

private:
    // create from joint
    OMJointState(RcsGraph* graph, RcsJoint* joint);

    // The graph being observed
    RcsGraph* graph;
    // The joint to observe
    RcsJoint* joint;
    // Set to true in order to wrap the joint angle into [-pi, pi].
    bool wrapJointAngle;
};

} /* namespace Rcs */

#endif /* _OMJOINTSTATE_H_ */
