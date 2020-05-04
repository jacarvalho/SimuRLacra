#ifndef _PHYSICSPARAMETERMANAGER_H_
#define _PHYSICSPARAMETERMANAGER_H_

#include "BodyParamInfo.h"
#include "PhysicsParameterDescriptor.h"
#include "../util/nocopy.h"

#include <PhysicsBase.h>

#include <vector>
#include <list>

namespace Rcs
{

/**
 * Main physics parameter modification system.
 * Keeps a list of parameter descriptors, allows setting values on them
 * and transfers those values to the physics simulation.
 *
 */
class PhysicsParameterManager
{
public:
    /**
     * Constructor.
     * @param graph graph to modify
     * @param physicsEngineName name of physics engine to use
     * @param physicsConfigFile config file for the physics engine
     */
    PhysicsParameterManager(
        RcsGraph* graph,
        const std::string& physicsEngineName,
        const std::string& physicsConfigFile);

    virtual ~PhysicsParameterManager();

    // not copy- or movable
    RCSPYSIM_NOCOPY_NOMOVE(PhysicsParameterManager)

    /**
     * Register a parameter descriptor operating on the given named body.
     * Takes ownership of the descriptor.
     * @param bodyName name of body to use
     * @param desc descriptor of the parameters to make changeable on the body.
     */
    void addParam(const char* bodyName, PhysicsParameterDescriptor* desc);

    /**
     * Get the BodyParamInfo object for the given named body, creating it if it doesn't exist.
     * @param bodyName name of body to look up.
     */
    BodyParamInfo* getBodyInfo(const char* bodyName);

    /**
     * Query current parameter values.
     */
    void getValues(PropertySink* outValues) const;

    /**
     * Create a new physics simulation using the given physics parameter values.
     * @param values parameter values to set
     * @return new physics simulator
     */
    PhysicsBase* createSimulator(PropertySource* values);

private:
    // graph to update
    RcsGraph* graph;

    // name of physics engine to use
    std::string physicsEngineName;

    // parsed physics engine configuration.
    PhysicsConfig* physicsConfig;

    // list of modifyable bodies. Uses std::list to get persistent references to elements.
    std::list<BodyParamInfo> bodyInfos;

    // list of parameter descriptors
    std::vector<PhysicsParameterDescriptor*> paramDescs;
};

} /* namespace Rcs */

#endif /* _PHYSICSPARAMETERMANAGER_H_ */
