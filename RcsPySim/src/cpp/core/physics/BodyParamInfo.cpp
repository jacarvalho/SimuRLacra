#include "BodyParamInfo.h"

#include <Rcs_typedef.h>

#include <algorithm>

Rcs::BodyParamInfo::BodyParamInfo(RcsGraph* graph, const char* bodyName, Rcs::PhysicsConfig* physicsConfig)
{
    this->graph = graph;
    this->body = RcsGraph_getBodyByName(graph, bodyName);

    // prefix = lowercase body name + _
    paramNamePrefix = bodyName;
    std::transform(paramNamePrefix.begin(), paramNamePrefix.end(), paramNamePrefix.begin(), ::tolower);
    paramNamePrefix += "_";

    // extract material from first physics shape
    RCSBODY_TRAVERSE_SHAPES(body) {
        if ((SHAPE->computeType & RCSSHAPE_COMPUTE_PHYSICS) != 0) {
            // found material-defining shape.
            // on vortex, there might be more materials, but that is not needed for now
            material = physicsConfig->getMaterial(SHAPE->material);
        }
    }


    modifiedFlag = 0;
}
