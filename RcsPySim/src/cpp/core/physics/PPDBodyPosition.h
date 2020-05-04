#ifndef _PPDBODYPOSITION_H_
#define _PPDBODYPOSITION_H_

#include "PPDCompound.h"

namespace Rcs
{

/**
 * Adjusts the Cartesian position of a body in space by adding an offset.
 * The individual dimensions can be masked out by passing false.
 */
class PPDBodyPosition : public PPDCompound
{
public:
  PPDBodyPosition(bool includeX, bool includeY, bool includeZ);

  ~PPDBodyPosition();

  virtual void setValues(PropertySource* inValues);

protected:
  virtual void init(BodyParamInfo* bpi);

private:
    //! The body's nominal position (in world coordinates / the parent's coordinates)
    double initPos[3];

    //! Offset
    double offset[3];
};

}

#endif //_PPDBODYPOSITION_H_
