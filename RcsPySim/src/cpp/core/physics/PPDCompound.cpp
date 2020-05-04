#include "PPDCompound.h"

namespace Rcs
{

PPDCompound::PPDCompound() = default;

PPDCompound::~PPDCompound()
{
    for (auto child: children)
    {
        delete child;
    }
}

void PPDCompound::getValues(PropertySink* outValues)
{
    for (auto child: children)
    {
        child->getValues(outValues);
    }
}

void PPDCompound::setValues(PropertySource* inValues)
{
    for (auto child: children)
    {
        child->setValues(inValues);
    }
}

void PPDCompound::addChild(PhysicsParameterDescriptor* child)
{
    children.push_back(child);
}

void PPDCompound::init(BodyParamInfo* bodyParamInfo)
{
    PhysicsParameterDescriptor::init(bodyParamInfo);
    for (auto child: children)
    {
        child->init(bodyParamInfo);
    }
}

const std::vector<PhysicsParameterDescriptor*>& PPDCompound::getChildren() const
{
    return children;
}

} /* namespace Rcs */
