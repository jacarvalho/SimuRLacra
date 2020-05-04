#ifndef _PPDSINGLEVAR_H_
#define _PPDSINGLEVAR_H_

#include "PhysicsParameterDescriptor.h"

#include <functional>

namespace Rcs
{

/**
 * Descriptor for a single scalar variable of type T.
 * The actual parameter name is built to be [lower case body name]_[param name].
 */
template<typename T>
class PPDSingleVar : public PhysicsParameterDescriptor
{
public:
    /*!
     * Returns a reference to the variable whihc this descriptor uses.
     */
    typedef std::function<T & (BodyParamInfo & )> VariableAccessor;

    /**
     * Constructor.
     * @param name unprefixed name for the parameter
     * @param modifiedFlag modified flag value to set for BodyParamInfo
     * @param variableAccessor variable accessor function. Returns a reference to be readable and writable.
     */
    PPDSingleVar(std::string name, int modifiedFlag, VariableAccessor variableAccessor) :
        name(std::move(name)), modifiedFlag(modifiedFlag), variableAccessor(variableAccessor) {}

    virtual ~PPDSingleVar() = default;

    virtual void getValues(PropertySink* outValues)
    {
        outValues->setProperty(name.c_str(), variableAccessor(*bodyParamInfo));
    }

    virtual void setValues(PropertySource* inValues)
    {
        // Get value reference
        T& ref = variableAccessor(*bodyParamInfo);

        // Try to get from dict
        bool set = inValues->getProperty(ref, name.c_str());

        // Set changed flag
        if (set)
        {
            bodyParamInfo->markChanged(modifiedFlag);
        }
    }

protected:
    virtual void init(BodyParamInfo* bodyParamInfo)
    {
        PhysicsParameterDescriptor::init(bodyParamInfo);
        name = bodyParamInfo->paramNamePrefix + name;
    }

private:
    //! Parameter name/key which the init() method will add a body prefix to this
    std::string name;

    //! Value to or to BodyParamInfo::modifiedFlag when modified
    int modifiedFlag;

    //! Variable accessor function
    VariableAccessor variableAccessor;
};


} /* namespace Rcs */

#endif /* _PPDSINGLEVAR_H_ */
