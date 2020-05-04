#ifndef _PPDCOMPOUND_H_
#define _PPDCOMPOUND_H_

#include "PhysicsParameterDescriptor.h"

#include <vector>

namespace Rcs
{

/**
 * Combines multiple child descriptors.
 */
class PPDCompound : public PhysicsParameterDescriptor
{
private:
    //! List of children
    std::vector<PhysicsParameterDescriptor*> children;

protected:
    virtual void init(BodyParamInfo* bodyParamInfo);

public:
    PPDCompound();

    virtual ~PPDCompound();

    /**
     * Register a child descriptor.
     * Takes ownership of the given object.
     */
    void addChild(PhysicsParameterDescriptor*);

    // Overridden to delegate to children
    virtual void getValues(PropertySink* outValues);

    // Overridden to delegate to children
    virtual void setValues(PropertySource* inValues);

    const std::vector<PhysicsParameterDescriptor*>& getChildren() const;
};

} /* namespace Rcs */

#endif /* _PPDCOMPOUND_H_ */
