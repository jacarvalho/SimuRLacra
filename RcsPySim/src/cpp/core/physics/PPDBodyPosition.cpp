#include "PPDBodyPosition.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>
#include <Rcs_body.h>
#include <Rcs_shape.h>
#include <Rcs_math.h>
#include <Rcs_Vec3d.h>

#include <stdexcept>

namespace Rcs
{


PPDBodyPosition::PPDBodyPosition(const bool includeX, const bool includeY, const bool includeZ)
{
    Vec3d_setZero(initPos);
    Vec3d_setZero(offset);

    if (includeX)
    {
        addChild(new PPDSingleVar<double>(
            "pos_offset_x", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[0]); })
            );
    }
    if (includeY)
    {
        addChild(new PPDSingleVar<double>(
            "pos_offset_y", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[1]); })
            );
    }
    if (includeZ)
    {
        addChild(new PPDSingleVar<double>(
            "pos_offset_z", BodyParamInfo::MOD_POSITION, [this](BodyParamInfo& bpi) -> double& { return (offset[2]); })
            );
    }

    if (getChildren().empty())
    {
        throw std::invalid_argument("No position specified for PPDBodyPosition!");
    }
}

PPDBodyPosition::~PPDBodyPosition() = default;

void PPDBodyPosition::init(Rcs::BodyParamInfo* bpi)
{
    PPDCompound::init(bpi);

    Vec3d_copy(initPos, bpi->body->A_BP->org);
}

void PPDBodyPosition::setValues(PropertySource* inValues)
{
    PPDCompound::setValues(inValues);

    // Apply the position offset to the body
    Vec3d_add(this->bodyParamInfo->body->A_BP->org, initPos, offset);
}

} /* namespace Rcs */
