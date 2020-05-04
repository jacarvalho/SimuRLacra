#include "AMPlateAngPos.h"

#include <Rcs_macros.h>
//#include <TaskPose6D.h>
#include <TaskPosition3D.h>
#include <TaskGenericEuler3D.h>

namespace Rcs
{

AMPlateAngPos::AMPlateAngPos(RcsGraph* graph) : ActionModelIK(graph)
{
    // lookup plate body on desired state graph
    RcsBody* plate = RcsGraph_getBodyByName(desiredGraph, "Plate");
    RCHECK(plate);
    // add the dynamicalSystems
    addTask(new TaskPosition3D(desiredGraph, plate, NULL, NULL));
    addTask(new TaskGenericEuler3D(desiredGraph, "CABr", plate, NULL, NULL));

    // create state matrix
    x_des = MatNd_create(getController()->getTaskDim(), 1);
    // init state with current
    getController()->computeX(x_des);
}

AMPlateAngPos::~AMPlateAngPos()
{
    MatNd_destroy(x_des);
}

unsigned int AMPlateAngPos::getDim() const
{
    return 2;
}

void AMPlateAngPos::getMinMax(double* min, double* max) const
{
    double maxAngle = 45 * M_PI / 180;
    min[0] = -maxAngle;
    min[1] = -maxAngle;
    max[0] = maxAngle;
    max[1] = maxAngle;
}

std::vector<std::string> AMPlateAngPos::getNames() const
{
    return {"a", "b"};
}

void AMPlateAngPos::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des,
                                   const MatNd* action, double dt)
{
    // copy actions into relevant parts of x_des
    x_des->ele[4] = action->ele[0]; // alpha
    x_des->ele[5] = action->ele[1]; // beta

    // use IK to compute q_des
    computeIK(q_des, q_dot_des, T_des, x_des, dt);
}

void AMPlateAngPos::reset()
{
    ActionModelIK::reset();
    // init state with current
    getController()->computeX(x_des);
}

void AMPlateAngPos::getStableAction(MatNd* action) const
{
    MatNd* x_curr = NULL;
    MatNd_create2(x_curr, getController()->getTaskDim(), 1);
    // compute current state
    getController()->computeX(x_curr);
    // export relevant parts of action
    action->ele[0] = x_curr->ele[4]; // alpha
    action->ele[1] = x_curr->ele[5]; // beta
    // cleanup
    MatNd_destroy(x_curr);
}

ActionModel *AMPlateAngPos::clone(RcsGraph *newGraph) const
{
    auto res = new AMPlateAngPos(newGraph);
    res->setupCollisionModel(collisionMdl);
    return res;
}

} /* namespace Rcs */

