#ifndef _PPDSPHERERADIUS_H_
#define _PPDSPHERERADIUS_H_

#include "PPDSingleVar.h"

namespace Rcs
{

/**
 * Descriptor for the radius of a sphere-shaped body.
 *
 * The sphere must be the first shape of the body for this to work.
 * This is specific to the ball-on-plate task since it also has to adjust
 * the ball's position to prevent it from clipping through the plate.
 *
 * Note that this does not update the inertia based on the shape changes,
 * for that, add PPDMassProperties after this descriptor.
 *
 * Exposes:
 *
 * - radius:
 *   Radius of the sphere.
 *   Unit: m
 */
class PPDSphereRadius : public PPDSingleVar<double>
{
public:
    PPDSphereRadius();

    virtual ~PPDSphereRadius();

    virtual void setValues(PropertySource* inValues);

protected:
    virtual void init(BodyParamInfo* bodyParamInfo);
};

} /* namespace Rcs */

#endif /* _PPDSPHERERADIUS_H_ */
