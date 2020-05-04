#include "PhysicsParameterManager.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>

#include <PhysicsFactory.h>

namespace Rcs
{

PhysicsParameterManager::PhysicsParameterManager(RcsGraph* graph,
                                                 const std::string& physicsEngineName,
                                                 const std::string& physicsConfigFile) :
                                                 graph(graph), physicsEngineName(physicsEngineName)
{
    physicsConfig = new PhysicsConfig(physicsConfigFile.c_str());
}

PhysicsParameterManager::~PhysicsParameterManager()
{
    for (auto pdesc : this->paramDescs) {
        delete pdesc;
    }
    delete physicsConfig;
}

void PhysicsParameterManager::addParam(const char* bodyName, PhysicsParameterDescriptor* desc)
{
    // obtain body info
    BodyParamInfo* bpi = getBodyInfo(bodyName);
    // init descriptor
    desc->init(bpi);
    // add it to list
    paramDescs.push_back(desc);
}

BodyParamInfo* PhysicsParameterManager::getBodyInfo(const char* bodyName)
{
    // check if used already
    for (BodyParamInfo& existing : bodyInfos)
    {
        if (STREQ(existing.body->name, bodyName))
        {
            return &existing;
        }
    }
    // not found, so add
    bodyInfos.emplace_back(graph, bodyName, physicsConfig);
    return &bodyInfos.back();
}


void PhysicsParameterManager::getValues(PropertySink* outValues) const
{
    for (auto pdesc : this->paramDescs)
    {
        pdesc->getValues(outValues);
    }
}

PhysicsBase* PhysicsParameterManager::createSimulator(PropertySource* values)
{
    for (auto pdesc : this->paramDescs)
    {
        pdesc->setValues(values);
    }

    return PhysicsFactory::create(physicsEngineName.c_str(), graph, physicsConfig);
}

} /* namespace Rcs */
