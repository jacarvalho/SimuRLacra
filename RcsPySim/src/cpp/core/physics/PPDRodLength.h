#ifndef _PPDRODLENGTH_H
#define _PPDRODLENGTH_H

#include "PhysicsParameterDescriptor.h"

namespace Rcs
{

/**
 * Adjusts a rod's / cylinder's length and shift it accordingly.
 * It assumes that the rod is aligned with the body's z axis.
 * This class has be written with the QuanserQube in mind
 * The mass properties are not adjusted automatically, so you should put a PPDMassProperties behind this descriptor.
 */
class PPDRodLength : public PhysicsParameterDescriptor
{
private:
    // Name of value to read
    std::string propertyName;
    // The rod shape
    RcsShape* rodShape;

    // Direction of rod, defaults to z
    double rodDirection[3];
    // Offset between rod start and body origin
    double rodOffset[3];

    // The child joint at the rod end. This is set for the arm body and used to adjust the pendulum joint position
    RcsJoint* childJoint;
    // Offset between joint origin and rod end, plus rodOffset. Thus the joint pos is jointOffset + rodDir * length.
    double jointOffset[3];

protected:
    virtual void init(BodyParamInfo* bpi);

public:
    PPDRodLength();

    virtual void getValues(PropertySink* outValues);

    virtual void setValues(PropertySource* inValues);
};

}

#endif //_PPDRODLENGTH_H
