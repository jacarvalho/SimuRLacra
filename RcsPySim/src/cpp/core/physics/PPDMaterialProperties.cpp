#include "PPDMaterialProperties.h"

#include <libxml/tree.h>

#include <array>

namespace Rcs
{

// Vortex extended property list
static const char * extended_xml_material_props[] = {
        "slip",
        "compliance",
};

PPDMaterialProperties::PPDMaterialProperties()
{}

PPDMaterialProperties::~PPDMaterialProperties() = default;


void PPDMaterialProperties::getValues(PropertySink* outValues)
{
    std::string prefixedName;
    // Bullet and Vortex
    prefixedName = bodyParamInfo->paramNamePrefix + "friction_coefficient";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getFrictionCoefficient());
    prefixedName = bodyParamInfo->paramNamePrefix + "rolling_friction_coefficient";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getRollingFrictionCoefficient());
    prefixedName = bodyParamInfo->paramNamePrefix + "restitution";
    outValues->setProperty(prefixedName.c_str(), bodyParamInfo->material.getRestitution());

    // Extension properties stored in the config xml-file
    for(auto propname : extended_xml_material_props) {
        auto configValue = xmlGetProp(bodyParamInfo->material.materialNode, BAD_CAST propname);
        if (configValue != NULL) {
            prefixedName = bodyParamInfo->paramNamePrefix + propname;
            // TODO the types aren't well defined here. We just assume string for now.
            outValues->setProperty(prefixedName.c_str(), configValue);
        }
    }
}

void PPDMaterialProperties::setValues(PropertySource* inValues)
{
    std::string prefixedName;
    double value;
    // Bullet and Vortex
    prefixedName = bodyParamInfo->paramNamePrefix + "friction_coefficient";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setFrictionCoefficient(value);
    }
    prefixedName = bodyParamInfo->paramNamePrefix + "rolling_friction_coefficient";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setRollingFrictionCoefficient(value);
    }
    prefixedName = bodyParamInfo->paramNamePrefix + "restitution";
    if (inValues->getProperty(value, prefixedName.c_str())) {
        bodyParamInfo->material.setRestitution(value);
    }

    // Extension properties stored in the config xml-file
    std::string configValue;
    for(auto propname : extended_xml_material_props) {
        prefixedName = bodyParamInfo->paramNamePrefix + propname;
        // Transfer config value as string to support all possible kinds
        if (inValues->getProperty(configValue, prefixedName.c_str())) {
            // Store in the xml-file
            xmlSetProp(bodyParamInfo->material.materialNode, BAD_CAST propname, BAD_CAST configValue.c_str());
        }
    }
}

} /* namespace Rcs */
