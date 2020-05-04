#ifndef _PPDMATERIALPROPERTIES_H_
#define _PPDMATERIALPROPERTIES_H_

#include "PhysicsParameterDescriptor.h"

namespace Rcs
{

/**
 * Descriptor for the body's friction and rolling friction coefficients as well as other material related properties.
 *
 * Exposes:
 *
 * - friction_coefficient
 *   Linear friction coefficient
 *   Unitless
 * - rolling_friction_coefficient
 *   Linear friction coefficient
 *   Unit: m, multiply unitless coefficient with contact surface curvature.
 *
 * Vortex only (see vortex documentation for details):
 * - slip
 * - compliance
 */
class PPDMaterialProperties : public PhysicsParameterDescriptor
{
public:
    PPDMaterialProperties();

    virtual ~PPDMaterialProperties();


    virtual void getValues(PropertySink* outValues);
    virtual void setValues(PropertySource* inValues);
};

} /* namespace Rcs */

#endif /* _PPDMATERIALPROPERTIES_H_ */
