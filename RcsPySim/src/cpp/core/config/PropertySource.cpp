#include "PropertySource.h"
#include "PropertySink.h"

#include <fstream>

namespace Rcs
{

PropertySource::PropertySource()
{
    // nothing to do
}

PropertySource::~PropertySource()
{
    // nothing to do
}


class EmptyPropertySource : public PropertySource {
public:
    virtual bool exists() {
        return false;
    }

    virtual bool getProperty(std::string& out, const char* property) {
        return false;
    }
    virtual bool getProperty(double& out, const char* property) {
        return false;
    }
    virtual bool getProperty(int& out, const char* property) {
        return false;
    }
    virtual bool getProperty(MatNd*& out, const char* property) {
        return false;
    }

    virtual bool getPropertyBool(const char* property, bool def = false) {
        return def;
    }
    virtual PropertySource* getChild(const char* prefix) {
        return empty();
    }
    virtual const std::vector<PropertySource*>& getChildList(const char* prefix) {
        static std::vector<PropertySource*> emptyList;
        return emptyList;
    }
    virtual PropertySource *clone() const {
        return empty();
    }

    virtual void saveXML(const char* fileName, const char* rootNodeName)
    {
        std::ofstream out;
        out.open(fileName);
        // no need to use libxml for writing an empty file
        out << "<?xml version=\"1.0\"?>" << std::endl;
        out << "<" << rootNodeName << " />" << std::endl;
    }
};


PropertySource* PropertySource::empty()
{
    static EmptyPropertySource emptySource;
    return &emptySource;
}

// These are just dummies, no extra file needed!

PropertySink::PropertySink()
{
    // Abstract base interface, it's empty!
}

PropertySink::~PropertySink()
{
    // Abstract base interface, it's empty!
}

} /* namespace Rcs */
