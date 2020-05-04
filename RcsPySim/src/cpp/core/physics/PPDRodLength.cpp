#include "PPDRodLength.h"

#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_body.h>
#include <Rcs_Vec3d.h>

Rcs::PPDRodLength::PPDRodLength() : rodShape(NULL), childJoint(NULL)
{
    // Initial rod direction is along the z-axis
    Vec3d_setUnitVector(rodDirection, 2);
    Vec3d_setZero(rodOffset);
    Vec3d_setZero(jointOffset);
}

void Rcs::PPDRodLength::init(Rcs::BodyParamInfo* bpi)
{
    PhysicsParameterDescriptor::init(bpi);

    propertyName = bpi->paramNamePrefix + "length";

    // Locate the rod shape
    RCSBODY_TRAVERSE_SHAPES(bpi->body)
    {
        if (SHAPE->type == RCSSHAPE_CYLINDER && (SHAPE->computeType & RCSSHAPE_COMPUTE_PHYSICS) != 0)
        {
            rodShape = SHAPE;
            break;
        }
    }
    RCHECK_MSG(rodShape, "No cylinder shape found on body %s.", bpi->body->name);

    // Get rod length as set in xml
    double initialRodLength = rodShape->extents[2];

    // The rod direction in shape frame is along the z-axis
    Vec3d_setUnitVector(rodDirection, 2);

    // Convert to body frame
    Vec3d_transRotateSelf(rodDirection, rodShape->A_CB.rot);

    // The rod has, per se, no real good way to determine if it's in the positive or negative direction.
    // however, we can guess by looking on which side of the body origin the rod center is
    double curDistToOrigin = Vec3d_innerProduct(rodDirection, rodShape->A_CB.org);
    if (curDistToOrigin < 0)
    {
        // the rod goes into -direction, so invert direction
        Vec3d_constMulSelf(rodDirection, -1);
    }

    // The rod offset is the difference between the rod's initial start and the body's origin
    // shapePos = offset + rodDir * length / 2
    // => offset = shapePos - rodDir * length / 2
    Vec3d_constMulAndAdd(rodOffset, rodShape->A_CB.org, rodDirection, -initialRodLength/2);

    if (STREQ(bpi->body->name, "Arm"))
    {
        // locate pendulum body as child if any
        RcsBody* pendulumChild = NULL;
        for (RcsBody* child = bpi->body->firstChild; child != NULL; child = child->next)
        {
            if (STREQ(child->name, "Pendulum"))
            {
                pendulumChild = child;
                break;
            }
        }
        RCHECK_MSG(pendulumChild, "Arm body doesn't have a pendulum child.");

        // Extract joint
        childJoint = pendulumChild->jnt;

        // Compute offset between rod end and joint if any
        Vec3d_constMulAndAdd(jointOffset, childJoint->A_JP->org, rodDirection, -initialRodLength);
    }
}

void Rcs::PPDRodLength::setValues(Rcs::PropertySource* inValues)
{
    double length;
    if (!inValues->getProperty(length, propertyName.c_str()))
    {
        // not set, ignore
        return;
    }

    // Set shape extends
    rodShape->extents[2] = length;
    // The shape origin is in the middle of the rod, the body origin at the end. Thus, shift the shape.
    Vec3d_constMulAndAdd(rodShape->A_CB.org, rodOffset, rodDirection, length/2);

    // Also adjust child joint if needed
    if (childJoint != NULL)
    {
        // the joint should be at the end of the rod.
        Vec3d_constMulAndAdd(childJoint->A_JP->org, jointOffset, rodDirection, length);
    }
}

void Rcs::PPDRodLength::getValues(PropertySink* outValues)
{
    outValues->setProperty(propertyName.c_str(), rodShape->extents[2]);
}
