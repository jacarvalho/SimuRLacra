#include "integrator.h"

#include <Rcs_macros.h>

namespace Rcs
{

void intStep2ndOrder(MatNd* x, MatNd* xd, const MatNd* xdd, double dt, IntMode mode)
{
    MatNd* aux_d = NULL, * aux_dd = NULL, * aux_sum = NULL;
    MatNd_fromStack(aux_d, x->m, 1);
    MatNd_fromStack(aux_dd, x->m, 1);
    MatNd_fromStack(aux_sum, x->m, 1);
    switch (mode) {
        case IntMode::ForwardEuler: // no energy loss noticed
            MatNd_constMul(aux_dd, xdd, 0.5 * dt * dt); // 0.5*xdd_0*dt^2
            MatNd_constMul(aux_d, xd, dt); // xd_0*dt
            MatNd_add(aux_sum, aux_d, aux_dd); // xd_0*dt + 0.5*xdd_0*dt^2
            MatNd_addSelf(x, aux_sum); // x_T = x_0 + dx_0*dt + 0.5*xdd_0^2*dt^2

            MatNd_constMulAndAddSelf(xd, xdd, dt); // xd_T = xd_0 + xdd_0*dt
            break;
        case IntMode::SymplecticEuler: // slight energy loss noticed

            MatNd_constMulAndAddSelf(xd, xdd, dt); // xd_T = xd_0 + xdd_0*dt

            MatNd_constMul(aux_sum, xd, dt); // xd_T*dt
            MatNd_addSelf(x, aux_sum); // x_T = x_0 + x_T*dt
            break;
        default:
            RFATAL("Invalid parameter value 'mode'!");
    }
}

void intStep1stOrder(MatNd* x, const MatNd* xd_0, const MatNd* xd_T, double dt, IntMode mode)
{
    switch (mode) {
        case IntMode::ForwardEuler:
            // x_T = x_0 + xd_0*dt
            MatNd_constMulAndAddSelf(x, xd_0, dt); // x_0 <- x_T (reuse of the variable)
            break;
        case IntMode::BackwardEuler:
            // x_T = x_0 + xd_T*dt
            MatNd_constMulAndAddSelf(x, xd_T, dt); // x_0 <- x_T (reuse of the variable)
            break;
        default:
            RFATAL("Invalid parameter value 'mode'!");
    }
}

}