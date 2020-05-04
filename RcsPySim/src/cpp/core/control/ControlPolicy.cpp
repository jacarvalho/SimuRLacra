#include "ControlPolicy.h"

#include "../config/PropertySource.h"

#include <Rcs_macros.h>
#include <Rcs_resourcePath.h>

#include <map>
#include <sstream>

namespace Rcs
{

// The policy type registry
static std::map<std::string, ControlPolicy::ControlPolicyCreateFunction> registry;

void ControlPolicy::registerType(const char* name,
        ControlPolicy::ControlPolicyCreateFunction creator)
{
    // Store in registry
    registry[name] = creator;
}

ControlPolicy* ControlPolicy::create(const char* name, const char* dataFile)
{
    // lookup factory for type
    auto iter = registry.find(name);
    if (iter == registry.end()) {
        std::ostringstream os;
        os << "Unknown control policy type '" << name << "'.";
        throw std::invalid_argument(os.str());
    }

    // find data file
    char filepath[256];
    bool found = Rcs_getAbsoluteFileName(dataFile, filepath);
    if (!found)
    {
        // file does not exist
        Rcs_printResourcePath();
        std::ostringstream os;
        os << "Policy file '" << dataFile << "' does not exist.";
        throw std::invalid_argument(os.str());
    }

    // Create instance
    return iter->second(filepath);
}

ControlPolicy* ControlPolicy::create(PropertySource* config)
{
    std::string policyType;
    std::string policyFile;
    RCHECK(config->getProperty(policyType, "type"));
    RCHECK(config->getProperty(policyFile, "file"));
    return Rcs::ControlPolicy::create(policyType.c_str(), policyFile.c_str());
}

std::vector<std::string> ControlPolicy::getTypeNames()
{
    std::vector<std::string> names;
    for (auto& elem : registry) {
        names.push_back(elem.first);
    }
    return names;
}

ControlPolicy::ControlPolicy()
{
    // Does nothing
}

ControlPolicy::~ControlPolicy()
{
    // Does nothing
}

void ControlPolicy::reset()
{
    // Does nothing by default
}

} /* namespace Rcs */
