#ifndef _PPDBOXEXTENTS_H_
#define _PPDBOXEXTENTS_H_

#include "PPDCompound.h"

namespace Rcs
{

/**
 * Adjusts the length width and high of a box shaped body.
 * This class does not consider potentially necessary offsets, e.g. if the box is initially in contact with another body.
 * The individual dimensions can be masked out by passing false.
 */
class PPDBoxExtents : public PPDCompound
{
public:
  PPDBoxExtents(unsigned int shapeIdx, bool includeLength, bool includeWidth, bool includeHeight);

  ~PPDBoxExtents();

  virtual void setValues(PropertySource* inValues);

protected:
  virtual void init(BodyParamInfo* bpi);

private:
    //! The shape's index given the body. THis is given by the order of the shapes in the config xml-file.
    unsigned int shapeIdx;

    //! The body's initial extents
    double initExtents[3];
};

}

#endif //_PPDBOXEXTENTS_H_
