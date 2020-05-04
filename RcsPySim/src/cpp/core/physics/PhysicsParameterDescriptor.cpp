#include "PhysicsParameterDescriptor.h"

namespace Rcs
{

PhysicsParameterDescriptor::PhysicsParameterDescriptor() :
        bodyParamInfo(NULL)
{
}

PhysicsParameterDescriptor::~PhysicsParameterDescriptor() = default;

void PhysicsParameterDescriptor::init(BodyParamInfo* bodyParamInfo)
{
    this->bodyParamInfo = bodyParamInfo;
}

} /* namespace Rcs */

