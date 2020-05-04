#ifndef SRC_CPP_CONFIG_PROPERTYSOURCEXML_H_
#define SRC_CPP_CONFIG_PROPERTYSOURCEXML_H_

#include "PropertySource.h"

#include <libxml/tree.h>

#include <map>

namespace Rcs
{

/**
 * Property source backed by the attributes of an xml node.
 */
class PropertySourceXml: public PropertySource
{
private:
    xmlNodePtr node;
    xmlDocPtr doc;

    std::map<std::string, PropertySource*> children;
    std::map<std::string, std::vector<PropertySource*>> listChildren;
public:
    /**
     * Constructor
     *
     * @param[in] node node whose attributes will be read
     * @param[in] doc  document to destroy on close. Use to make the node owned by this class. If null, node will not be destroyed.
     */
    PropertySourceXml(xmlNodePtr node,  xmlDocPtr doc = NULL);

    /**
     * Constructor, loading XML from filename.
     * Expects a root tag named 'Experiment'.
     * @param configFile xml file name
     */
    explicit PropertySourceXml(const char* configFile);
    virtual ~PropertySourceXml();

    virtual bool exists();

    virtual bool getProperty(std::string& out, const char* property);
    virtual bool getProperty(double& out, const char* property);
    virtual bool getProperty(int& out, const char* property);
    virtual bool getProperty(MatNd*& out, const char* property);

    virtual bool getPropertyBool(const char* property, bool def = false);

    virtual PropertySource* getChild(const char* prefix);

    virtual const std::vector<PropertySource*>& getChildList(const char* prefix);

    virtual PropertySource* clone() const;

    virtual void saveXML(const char* fileName, const char* rootNodeName);
};

} /* namespace Rcs */

#endif /* SRC_CPP_CONFIG_PROPERTYSOURCEXML_H_ */
