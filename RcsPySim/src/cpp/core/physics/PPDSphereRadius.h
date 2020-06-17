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
 *   Radius of the sphere [m]
 */
class PPDSphereRadius : public PPDSingleVar<double>
{
public:
    /**
     * Constructor
     *
     * @param shapeIdx The spheres's index within given the body.
     *                 This is given by the order of the shapes in the config xml-file.
     * @param prevBodyName Name of the previous body if the graph to which the sphere is placed relative to,
     *                     Use "" if the sphere is defined in world coordinates
     */
    PPDSphereRadius(std::string prevBodyName, unsigned int shapeIdx = 0, unsigned int shapeIdxPrevBody = 0);

    virtual ~PPDSphereRadius();

    virtual void setValues(PropertySource* inValues);

protected:
    virtual void init(BodyParamInfo* bodyParamInfo);

private:
    //! Name of the previous body if the graph to which the sphere is placed relative to
    std::string prevBodyName;

    //! The spheres's index within given the body. This is given by the order of the shapes in the config xml-file.
    unsigned int shapeIdx;
    //! The spheres's index within the previous body. This is given by the order of the shapes in the config xml-file.
    unsigned int shapeIdxPrevBody;
};

} /* namespace Rcs */

#endif /* _PPDSPHERERADIUS_H_ */
