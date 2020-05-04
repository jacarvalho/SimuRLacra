#ifndef SRC_CPP_CONFIG_PROPERTYSOURCE_H_
#define SRC_CPP_CONFIG_PROPERTYSOURCE_H_

#include <Rcs_MatNd.h>

#include <string>
#include <vector>

namespace Rcs
{

/**
 * Base class for a source of configuration properties.
 * Basically stores values of different types by name.
 */
class PropertySource
{
public:
    PropertySource();
    virtual ~PropertySource();

    /**
     * Check if this property source exists in the underlying storage.
     */
    virtual bool exists() = 0;
    
    /**
     * Read a string value.
     *
     * @param[out] out      storage for read value
     * @param[in]  property name of property to read
     * @return true if the property was read successfully
     * @throws std::exception if the property exists, but couldn't be converted
     */
    virtual bool getProperty(std::string& out, const char* property) = 0;
    
    /**
     * Read a double value.
     *
     * @param[out] out      storage for read value
     * @param[in]  property name of property to read
     * @return true if the property was read successfully, false if it doesn't exist
     * @throws std::exception if the property exists, but couldn't be converted
     */
    virtual bool getProperty(double& out, const char* property) = 0;
    
    /**
     * Read an int value.
     *
     * @param[out] out      storage for read value
     * @param[in]  property name of property to read
     * @return true if the property was read successfully, false if it doesn't exist
     * @throws std::exception if the property exists, but couldn't be converted
     */
    virtual bool getProperty(int& out, const char* property) = 0;
    
    /**
     * Read a vector/matrix value.
     * The variable out should be a pointer and will be set to a newly created MatNd* on success.
     *
     * @param[out] out      storage for read value
     * @param[in]  property name of property to read
     * @return true if the property was read successfully, false if it doesn't exist
     * @throws std::exception if the property exists, but couldn't be converted
     */
    virtual bool getProperty(MatNd*& out, const char* property) = 0;
    
    
    /**
     * Read a boolean value, returning a default value if not found.
     * This interface differs from the others to ease usability
     *
     * @param[in]  property name of property to read
     * @param[in]  def      value to return if property doesn't exist
     * @return true if the property was read successfully
     * @throws std::exception if the property exists, but couldn't be converted
     */
    virtual bool getPropertyBool(const char* property, bool def = false) = 0;

    /**
     * Obtain a child property source.
     *
     * The exact meaning of this depends on the implementation. For an Xml document, it could be a child element.
     * For a python dict, it could be a dict-typed entry.
     *
     * The returned object is owned by the parent.
     */
    virtual PropertySource* getChild(const char* prefix) = 0;

    /**
     * Obtain a list of child property sources.
     *
     * The exact meaning of this depends on the implementation. For an Xml document, it could be child elements with the same tag name.
     * For a python dict, it could be a list of dicts.
     *
     * The returned objects are owned by the parent.
     */
    virtual const std::vector<PropertySource*>& getChildList(const char* prefix) = 0;

    /**
     * Create an independent copy of this property source.
     * @return a copy of this property source. Must take ownership.
     */
    virtual PropertySource* clone() const = 0;
  
  /**
   * Save this property source as xml file.
   * @param[in] fileName name of the file to write
   * @param[in] rootNodeName name of the xml root node
   */
    virtual void saveXML(const char* fileName, const char* rootNodeName) = 0;

    /**
     * A singleton property source without entries.
     */
    static PropertySource* empty();
};

} /* namespace Rcs */

#endif /* SRC_CPP_CONFIG_PROPERTYSOURCE_H_ */
