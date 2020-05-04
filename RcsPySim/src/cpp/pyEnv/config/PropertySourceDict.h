#ifndef SRC_CPP_CONFIG_PROPERTYSOURCEDICT_H_
#define SRC_CPP_CONFIG_PROPERTYSOURCEDICT_H_

#include <config/PropertySource.h>
#include <config/PropertySink.h>

#include <pybind11/pybind11.h>

#include <map>

namespace Rcs
{

/**
 * PropertySource/Sink backed by a C++ dict.
 */
class PropertySourceDict: public PropertySink
{
private:
    pybind11::dict dict;

    std::map<std::string, PropertySourceDict*> children;
    std::map<std::string, std::vector<PropertySource*>> listChildren;

    // for lazy parent writing
    PropertySourceDict* parent;
    const char* prefix;
    bool _exists;

    // create child
    PropertySourceDict(
            pybind11::dict dict,
            PropertySourceDict* parent,
            const char* prefix,
            bool exists
    );
public:
    PropertySourceDict(pybind11::dict dict);
    virtual ~PropertySourceDict();

    virtual bool exists();

    virtual bool getProperty(std::string& out, const char* property);
    virtual bool getProperty(double& out, const char* property);
    virtual bool getProperty(int& out, const char* property);
    virtual bool getProperty(MatNd*& out, const char* property);

    virtual bool getPropertyBool(const char* property, bool def = false);

    virtual PropertySink* getChild(const char* prefix);

    // note: not editable for now
    virtual const std::vector<PropertySource*>& getChildList(const char* prefix);

    virtual void setProperty(const char* property,
            const std::string& value);
    virtual void setProperty(const char* property, bool value);
    virtual void setProperty(const char* property, int value);
    virtual void setProperty(const char* property, double value);
    virtual void setProperty(const char* property, MatNd* value);

    virtual PropertySink *clone() const;

    virtual void saveXML(const char* fileName, const char* rootNodeName);

protected:
    void onWrite();

    void appendPrefix(std::ostream&);
};

} /* namespace Rcs */

#endif /* SRC_CPP_CONFIG_PROPERTYSOURCEDICT_H_ */
