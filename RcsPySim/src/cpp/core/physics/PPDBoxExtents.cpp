#include "PPDBoxExtents.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>

#include <stdexcept>

namespace Rcs
{

#define DEF_EXTENT_PARAM(name, modflag, var) \
    addChild(new PPDSingleVar<double>((name), (modflag), [this](BodyParamInfo& bpi) -> double& {return (var);}))

PPDBoxExtents::PPDBoxExtents(
    unsigned int shapeIdx,
    const bool includeLength,
    const bool includeWidth,
    const bool includeHeight) : shapeIdx(shapeIdx)
{
    // Add the children of type PPDSingleVar
    if (includeLength)
    {
        DEF_EXTENT_PARAM("length", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[0]);
    }
    if (includeWidth)
    {
        DEF_EXTENT_PARAM("width", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[1]);
    }
    if (includeHeight)
    {
        DEF_EXTENT_PARAM("height", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[2]);
    }

    if (getChildren().empty())
    {
        throw std::invalid_argument("No position specified for PPDBoxExtents!");
    }
}

PPDBoxExtents::~PPDBoxExtents() = default;

void PPDBoxExtents::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);

    if (bpi->body->shape[this->shapeIdx]->type != RCSSHAPE_TYPE::RCSSHAPE_BOX)
    {
        throw std::invalid_argument("Using the PPDBoxExtents on a non-box shape!");
    }
}

void PPDBoxExtents::setValues(PropertySource* inValues)
{
    // Change the shape via the childrens' setValues() method without any other adaption
    PPDCompound::setValues(inValues);
}


PPDCubeExtents::PPDCubeExtents(unsigned int shapeIdx) : shapeIdx(shapeIdx)
{
    // Add the children of type PPDSingleVar
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[0]);
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[1]);
    DEF_EXTENT_PARAM("size", BodyParamInfo::MOD_SHAPE, bpi.body->shape[this->shapeIdx]->extents[2]);

    if (getChildren().empty())
    {
        throw std::invalid_argument("No position specified for PPDBoxExtents!");
    }
}

PPDCubeExtents::~PPDCubeExtents() = default;

void PPDCubeExtents::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);

    if (bpi->body->shape[this->shapeIdx]->type != RCSSHAPE_TYPE::RCSSHAPE_BOX)
    {
        throw std::invalid_argument("Using the PPDCubeExtents on a non-box shape!");
    }
}

void PPDCubeExtents::setValues(PropertySource* inValues)
{
    // Change the shape via the childrens' setValues() method without any other adaption
    PPDCompound::setValues(inValues);
}


} /* namespace Rcs */
