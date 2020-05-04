#ifndef SRC_CPP_CORE_CONFIG_PROPERTYSINK_H_
#define SRC_CPP_CORE_CONFIG_PROPERTYSINK_H_

#include "PropertySource.h"

#include <Rcs_MatNd.h>

namespace Rcs
{

/**
 * Mutable version of PropertySource.
 *
 * Note that this class does not specify any mechanism for persisting the changes made here.
 */
class PropertySink : public PropertySource
{
public:
    PropertySink();
    virtual ~PropertySink();

    /**
     * Set a property to the given string value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, const std::string& value) = 0;
    /**
     * Set a property to the given boolean value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, bool value) = 0;
    /**
     * Set a property to the given integer value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, int value) = 0;
    /**
     * Set a property to the given double value.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, double value) = 0;
    /**
     * Set a property to the given vector/matrix value.
     *
     * If one of the dimensions of the matrix has size 1,
     * it is interpreted as vector.
     *
     * @param[in] property name of property to write.
     * @param[in] value    new property value.
     */
    virtual void setProperty(const char* property, MatNd* value) = 0;

    /**
     * Obtain a child property sink.
     *
     * This only adapts the return type from the PropertySource definition.
     */
    virtual PropertySink* getChild(const char* prefix) = 0;

    // only adapt the return type
    virtual PropertySink* clone() const = 0;
};

} /* namespace Rcs */

#endif /* SRC_CPP_CORE_CONFIG_PROPERTYSINK_H_ */
