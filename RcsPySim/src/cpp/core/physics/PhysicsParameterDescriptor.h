#ifndef _PHYSICSPARAMETERDESCRIPTOR_H_
#define _PHYSICSPARAMETERDESCRIPTOR_H_

#include "BodyParamInfo.h"
#include "../config/PropertySource.h"
#include "../config/PropertySink.h"

namespace Rcs
{

/**
 * Descriptor for one or more physical parameters settable from Python.
 *
 * The parameters should be stored on the BodyParamInfo reference.
 */
class PhysicsParameterDescriptor
{
protected:
    // body to set parameters on
    BodyParamInfo* bodyParamInfo;
public:
    PhysicsParameterDescriptor();

    virtual ~PhysicsParameterDescriptor();

    /**
     * Read values from graph and put them into the given dict.
     */
    virtual void getValues(PropertySink* outValues) = 0;

    /**
     * Read values from the given dict and apply them to the graph.
     * The parameter names need to be the same as in Rcs, e.g. rolling_friction_coefficient.
     */
    virtual void setValues(PropertySource* inValues) = 0;

protected:

    friend class PhysicsParameterManager;

    friend class PPDCompound;

    /**
     * Setup descriptor to work on the given body reference.
     * Override for more custom initialization.
     */
    virtual void init(BodyParamInfo* bodyParamInfo);
};

} /* namespace Rcs */

#endif /* _PHYSICSPARAMETERDESCRIPTOR_H_ */
