#include "PPDMassProperties.h"
#include "PPDSingleVar.h"

#include <Rcs_typedef.h>
#include <Rcs_body.h>
#include <Rcs_shape.h>
#include <Rcs_math.h>

namespace Rcs
{

#define DEF_MASS_PARAM(name, modflag, var) \
    addChild(new PPDSingleVar<double>((name), (modflag), [](BodyParamInfo& bpi) -> double& {return (var);}))

PPDMassProperties::PPDMassProperties()
{
    DEF_MASS_PARAM("mass", BodyParamInfo::MOD_MASS, bpi.body->m);

    DEF_MASS_PARAM("com_x", BodyParamInfo::MOD_COM, bpi.body->Inertia->org[0]);
    DEF_MASS_PARAM("com_y", BodyParamInfo::MOD_COM, bpi.body->Inertia->org[1]);
    DEF_MASS_PARAM("com_z", BodyParamInfo::MOD_COM, bpi.body->Inertia->org[2]);

    DEF_MASS_PARAM("i_xx", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[0][0]);
    DEF_MASS_PARAM("i_xy", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[0][1]);
    DEF_MASS_PARAM("i_xz", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[0][2]);
    DEF_MASS_PARAM("i_yy", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[1][1]);
    DEF_MASS_PARAM("i_yz", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[1][2]);
    DEF_MASS_PARAM("i_zz", BodyParamInfo::MOD_INERTIA, bpi.body->Inertia->rot[2][2]);
}

PPDMassProperties::~PPDMassProperties() = default;


void PPDMassProperties::setValues(PropertySource* inValues)
{
    PPDCompound::setValues(inValues);

    if (bodyParamInfo->body->m == 0.0)
    {
        return;
    }
    double v_bdy = RcsBody_computeVolume(bodyParamInfo->body);
    if (v_bdy == 0.0)
    {
        return;
    }

    if (!bodyParamInfo->isChanged(BodyParamInfo::MOD_COM))
    {
        // Recompute the CoM from the shape since that function is not exported
        Vec3d_setZero(bodyParamInfo->body->Inertia->org);

        RCSBODY_TRAVERSE_SHAPES(bodyParamInfo->body)
        {
            double r_com_sh[3], v_sh;
            v_sh = RcsShape_computeVolume(SHAPE);
            RcsShape_computeLocalCOM(SHAPE, r_com_sh);
            Vec3d_constMulSelf(r_com_sh, v_sh/v_bdy);
            Vec3d_addSelf(bodyParamInfo->body->Inertia->org, r_com_sh);
        }
    }

    if (!bodyParamInfo->isChanged(BodyParamInfo::MOD_INERTIA))
    {
        Mat3d_setZero(bodyParamInfo->body->Inertia->rot);
        // Recompute the inertia from shapes using existing COM
        double density = bodyParamInfo->body->m/v_bdy;   // Density: rho = m / V

        RCSBODY_TRAVERSE_SHAPES(bodyParamInfo->body)
        {
            // Compute the shape's inertia around its COM (it is represented in the shape's frame of reference)
            double I_s[3][3];
            RcsShape_computeInertiaTensor(SHAPE, density, I_s);

            // Rotate inertia tensor from the shape frame into the body frame:
            // B_I = A_BC^T C_I A_CB
            Mat3d_similarityTransform(I_s, (double (*)[3]) SHAPE->A_CB.rot, I_s);

            // Add the Steiner term related to gravity center of shape in body frame
            double r_b_sgc[3];   // vector from body origin to shape gravity center
            RcsShape_computeLocalCOM(SHAPE, r_b_sgc);
            double r_sgc_bgc[3];   // vector from shape COM to body COM
            Vec3d_sub(r_sgc_bgc, bodyParamInfo->body->Inertia->org, r_b_sgc);
            double m_sh = density*RcsShape_computeVolume(SHAPE);
            Math_addSteinerToInertia(I_s, r_sgc_bgc, m_sh);
            Mat3d_addSelf(bodyParamInfo->body->Inertia->rot, I_s);
        }
    }
    else
    {
        // Fill up symmetric elements of inertia
        bodyParamInfo->body->Inertia->rot[1][0] = bodyParamInfo->body->Inertia->rot[0][1];
        bodyParamInfo->body->Inertia->rot[2][0] = bodyParamInfo->body->Inertia->rot[0][2];
        bodyParamInfo->body->Inertia->rot[2][1] = bodyParamInfo->body->Inertia->rot[1][2];
    }
}

} /* namespace Rcs */
