#include "PropertySourceXml.h"

#include <Rcs_parser.h>
#include <Rcs_stlParser.h>
#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

#include <stdexcept>
#include <sstream>

namespace Rcs
{


PropertySourceXml::PropertySourceXml(const char* configFile)
{
    // Determine absolute file name of config file and copy the XML file name
    char filename[256] = "";
    bool fileExists = Rcs_getAbsoluteFileName(configFile, filename);

    if (!fileExists)
    {
        RMSG("Resource path is:");
        Rcs_printResourcePath();
        RFATAL("Experiment configuration file \"%s\" not found in "
               "ressource path - exiting", configFile ? configFile : "NULL");
    }

    // load xml tree
    node = parseXMLFile(filename, "Experiment", &doc);
}

PropertySourceXml::PropertySourceXml(xmlNodePtr node, xmlDocPtr doc) :
        node(node), doc(doc)
{
    RCHECK(node != NULL);
}

PropertySourceXml::~PropertySourceXml()
{
    // delete children where needed
    for(auto& it: children) {
        if (it.second != empty()) {
            delete it.second;
        }
    }
    for(auto& it: listChildren) {
        for (auto le : it.second) {
            delete le;
        }
    }
    // delete doc if any
    if (doc != NULL) {
        xmlFreeDoc(doc);
    }
}

bool PropertySourceXml::exists()
{
    // will return empty() if it doesn't exist
    return true;
}

bool PropertySourceXml::getProperty(std::string& out, const char* property)
{
    // check if exists
    if (!getXMLNodeProperty(node, property)) {
        return false;
    }
    // get value string
    getXMLNodePropertySTLString(node, property, out);
    return true;
}

bool PropertySourceXml::getProperty(double& out, const char* property)
{
    // check if exists
    if (!getXMLNodeProperty(node, property)) {
        return false;
    }
    // read it
    if (!getXMLNodePropertyDouble(node, property, &out)) {
        std::ostringstream os;
        os << "Invalid double value: ";
        std::string value;
        getXMLNodePropertySTLString(node, property, value);
        os << value;
        throw std::invalid_argument(os.str());
    }
    return true;
}

bool PropertySourceXml::getProperty(int& out, const char* property)
{
    // check if exists
    if (!getXMLNodeProperty(node, property)) {
        return false;
    }
    // read it
    if (!getXMLNodePropertyInt(node, property, &out)) {
        std::ostringstream os;
        os << "Invalid double value: ";
        std::string value;
        getXMLNodePropertySTLString(node, property, value);
        os << value;
        throw std::invalid_argument(os.str());
    }
    return true;
}

bool PropertySourceXml::getProperty(MatNd*& out, const char* property)
{
    // check if exists
    if (!getXMLNodeProperty(node, property)) {
        return false;
    }
    // split into stl vector first
    std::vector<std::string> entries;
    getXMLNodePropertyVecSTLString(node, property, entries);

    // create output matrix using entry count
    out = MatNd_create(entries.size(), 1);

    // convert entries
    for (std::size_t i = 0; i < entries.size(); ++i) {
        // try parsing locale independently
        std::istringstream is(entries[i]);
        is.imbue(std::locale("C"));

        if(!(is >> out->ele[i])) {
            // invalid format

            // be sure to destroy the allocated storage before throwing
            MatNd_destroy(out);
            out = NULL;

            std::ostringstream os;
            os << "Invalid matrix entry value # " << i << ": " << entries[i];
            throw std::invalid_argument(os.str());
        }
    }
    return true;
}

bool PropertySourceXml::getPropertyBool(const char* property, bool def)
{
    // check if exists
    if (!getXMLNodeProperty(node, property)) {
        return def;
    }
    // read it
    bool res;
    getXMLNodePropertyBoolString(node, property, &res);
    return res;
}

PropertySource* PropertySourceXml::getChild(const char* prefix)
{
    std::string prefixStr = prefix;
    // check if it exists already
    auto iter = children.find(prefixStr);
    if (iter != children.end()) {
        return iter->second;
    }
    // try to get child node
    xmlNodePtr childNode = getXMLChildByName(node, prefix);

    PropertySource* result;
    if (childNode == NULL) {
        result = empty();
    } else {
        result = new PropertySourceXml(childNode);
    }
    children[prefixStr] = result;
    return result;
}

const std::vector<PropertySource*>& PropertySourceXml::getChildList(
        const char* prefix)
{
    std::string prefixStr = prefix;
    // check if it exists already
    auto iter = listChildren.find(prefixStr);
    if (iter != listChildren.end()) {
        return iter->second;
    }
    // create new entry
    auto& list = listChildren[prefixStr];

    // find matching children
    xmlNodePtr child = node->children;
    while(child) {
        if (STRCASEEQ((const char*) BAD_CAST child->name, prefix)) {
            list.push_back(new PropertySourceXml(child));
        }
        child = child->next;
    }

    return list;
}

PropertySource *PropertySourceXml::clone() const {
    // copy subtree recursively, store in new doc.
    xmlDocPtr cpdoc = xmlNewDoc(NULL);
    xmlNodePtr cpnode = xmlDocCopyNode(node, cpdoc, 1);
    xmlDocSetRootElement(cpdoc, cpnode);

    return new PropertySourceXml(node, doc);
}

void PropertySourceXml::saveXML(const char* fileName, const char* rootNodeName)
{
    // simple, we are already xml
    // might need to create a temporary doc if we are a sub node
    xmlDocPtr writeDoc = doc;
    if (writeDoc == NULL || !isXMLNodeName(node, rootNodeName))
    {
        // use a temporary doc for writing
        writeDoc  = xmlNewDoc(NULL);
        xmlNodePtr cpnode = xmlDocCopyNode(node, writeDoc, 1);
        xmlNodeSetName(cpnode, BAD_CAST rootNodeName);
        xmlDocSetRootElement(writeDoc, cpnode);
    }

    // perform save
    xmlIndentTreeOutput = 1;
    xmlSaveFormatFile(fileName, writeDoc, 1);

    // delete temp doc if any
    if (writeDoc != doc)
    {
        xmlFreeDoc(writeDoc);
    }
}

} /* namespace Rcs */
