#ifndef _PPDMASSPROPERTIES_H_
#define _PPDMASSPROPERTIES_H_

#include "PPDCompound.h"

namespace Rcs
{

/**
 * Descriptor for the body's mass, center of mass and inertia.
 *
 * If the center of mass or the inertia are not set, they will be calculated automatically.
 * Setting one part of com or inertia is enough to define them as set.
 *
 * Exposes:
 *
 * - mass
 *   Mass of the body
 *   Unit: kg
 * - com_x, com_y, com_z
 *   3d position of the center of mass
 *   Unit: m
 * - i_xx, i_xy, i_xz, i_yy, i_yz, i_zz
 *   Components of the inertia tensor
 *   Unit: kg m^2
 */
class PPDMassProperties : public PPDCompound
{
public:
    PPDMassProperties();

    virtual ~PPDMassProperties();

    virtual void setValues(PropertySource* inValues);
};

} /* namespace Rcs */

#endif /* _PPDMASSPROPERTIES_H_ */
