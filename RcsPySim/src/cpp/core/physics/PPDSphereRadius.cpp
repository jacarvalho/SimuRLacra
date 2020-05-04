#include "PPDSphereRadius.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

namespace Rcs
{

PPDSphereRadius::PPDSphereRadius() :
        PPDSingleVar("radius", BodyParamInfo::MOD_SHAPE,
                     // Use a lambda to access the variable.
                     [](BodyParamInfo& bpi) -> double& { return bpi.body->shape[0]->extents[0]; })
{
}

PPDSphereRadius::~PPDSphereRadius() = default;

void PPDSphereRadius::setValues(PropertySource* inValues)
{
    // adapt properties
    PPDSingleVar::setValues(inValues);

    // Adapt the ball's z-position to prevent clipping into the plate
    double newRad = bodyParamInfo->body->shape[0]->extents[0];

    // need to check if the ball position variable is relative
    RcsBody* plate = RcsGraph_getBodyByName(bodyParamInfo->graph, "Plate");
    double plateZ;
    if (bodyParamInfo->body->parent == plate) {
        // ball rigid body coordinates are relative, no offset needed
        plateZ = 0;
    } else {
        // ball ball rigid body coordinates are absolute, shift so that the ball is on the plate
        // NOTE: this assumes a level plate
        plateZ = plate->A_BI->org[2];
    }

    bodyParamInfo->graph->q->ele[bodyParamInfo->body->jnt->jointIndex + 2] = plateZ + newRad; // set ball's z DoF
    // Make sure the state is propagated
    RcsGraph_setState(bodyParamInfo->graph, NULL, bodyParamInfo->graph->q_dot);

    RLOG(4, "New radius = %f; New z-position = %f", newRad, plateZ + newRad);
}

void PPDSphereRadius::init(BodyParamInfo* bodyParamInfo)
{
    PPDSingleVar::init(bodyParamInfo);
    // check that the ball is valid
    RCHECK_MSG(bodyParamInfo->body->shape != NULL, "Invalid ball body %s", bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->shape[0] != NULL, "Invalid ball body %s", bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->shape[0]->type == RCSSHAPE_SPHERE, "Invalid ball body %s",
               bodyParamInfo->body->name);
    RCHECK_MSG((bodyParamInfo->body->shape[0]->computeType & RCSSHAPE_COMPUTE_PHYSICS) != 0, "Invalid ball body %s",
               bodyParamInfo->body->name);
    RCHECK_MSG(bodyParamInfo->body->rigid_body_joints, "Invalid ball body %s", bodyParamInfo->body->name);
}

} /* namespace Rcs */
