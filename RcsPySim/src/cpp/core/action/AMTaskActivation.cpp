#include "AMTaskActivation.h"
#include "ActionModelIK.h"
#include "../util/eigen_matnd.h"

#include <Rcs_macros.h>

#include <utility>


/*! Compute the column-wise softmax \f$ \sigma(x)_{i,j} = \frac{e^{x_{i,j}}}{\sum_{k=1}^{K} e^{x_{k,j}}} \f$.
 * The entries of the resulting matrix sum to one
 * @param[in]  src input matrix
 * @param[in]  beta scaling factor, as beta goes to infinity we get the argmax
 * @param[out] dst output matrix
 */
void MatNd_softMax(MatNd* dst, const MatNd* src, double beta)
{
    RCHECK_MSG((dst->m == src->m) && (dst->n == src->n), "dst: [%d x %d]  src: [%d x %d]", dst->m, dst->n, src->m,
               src->n);
    RCHECK_MSG(beta > 0, "beta: %f", beta);

    for (unsigned int j = 0; j < src->n; j++)
    {
        double dnom = 0.;
        for (unsigned int i = 0; i < src->m; i++)
        {
            dnom += exp(beta*MatNd_get2(src, i, j));
        }

        for (unsigned int i = 0; i < src->m; i++)
        {
            MatNd_set2(dst, i, j, exp(beta*MatNd_get2(src, i, j))/dnom);
        }
    }
}

/*! Find all unique combinations of 0s and 1s for N binary events.
 * @param[out] allComb 2^N x N output matrix containing the combinations,
 *             with N being the number of events that could be 0 or 1
 */
void findAllBitCombinations(MatNd* allComb)
{
    // Get the dimension
    size_t N = allComb->n;

    // Set all entries to 0. The first row will be used directly as a result
    MatNd_setZero(allComb);

    size_t i = 1;  // row index
    for (size_t k = 1; k <= N; k++)
    {
        std::vector<size_t> vec(N, 0);
        std::vector<size_t> currOnes(k, 1);
        // Set last k bits to 1
        std::copy(begin(currOnes), end(currOnes), end(vec) - k);

        // Get the combinations / permutations and set them into the matrix
        do
        {
            for (size_t v = 0; v < vec.size(); v++)
            {
                // Fill the current row
                MatNd_set2(allComb, i, v, vec[v]);
            }
            i++;
        } while (std::next_permutation(vec.begin(), vec.end()));  // returns false if no valid permutation was found
    }
}

namespace Rcs
{

AMTaskActivation::AMTaskActivation(ActionModel* wrapped, std::vector<DynamicalSystem*> ds, TaskCombinationMethod tcm)
    : ActionModel(wrapped->getGraph()), wrapped(wrapped), dynamicalSystems(std::move(ds)), taskCombinationMethod(tcm)
{
    activation = MatNd_create((unsigned int) dynamicalSystems.size(), 1);
}

AMTaskActivation::~AMTaskActivation()
{
    delete wrapped;
    delete activation;
    for (auto* ds : dynamicalSystems)
    {
        delete ds;
    }
}

unsigned int AMTaskActivation::getDim() const
{
    return (unsigned int) dynamicalSystems.size();
}

void AMTaskActivation::getMinMax(double* min, double* max) const
{
    // All activations are between -1 and 1
    for (unsigned int i = 0; i < getDim(); i++)
    {
        min[i] = -1;
        max[i] = 1;
    }
}

std::vector<std::string> AMTaskActivation::getNames() const
{
    std::vector<std::string> names;
    for (unsigned int i = 0; i < getDim(); ++i)
    {
        names.push_back("a_" + std::to_string(i));
    }
    return names;
}

void AMTaskActivation::computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt)
{
    RCHECK(action->n == 1);  // actions are column vectors

    // Remember x_dot from last step as integration input.
    Eigen::VectorXd x_dot_old = x_dot;
    x_dot.setConstant(0);

    // Collect data from each DS
    for (unsigned int i = 0; i < getDim(); ++i)
    {
        // Fill a temp x_dot with the old x_dot
        Eigen::VectorXd x_dot_ds = x_dot_old;

        // Step the DS
        dynamicalSystems[i]->step(x_dot_ds, x, dt);
        // Remember x_dot for OMDynamicalSystemDiscrepancy
        dynamicalSystems[i]->x_dot_des = x_dot_ds;

        // Combine the individual x_dot of every DS
        switch (taskCombinationMethod)
        {
            case TaskCombinationMethod::Sum:
            case TaskCombinationMethod::Mean:
            {
                x_dot += action->ele[i]*x_dot_ds;
                MatNd_set(activation, i, 0, action->ele[i]);
                break;
            }

            case TaskCombinationMethod::SoftMax:
            {
                MatNd* a = NULL;
                MatNd_create2(a, action->m, action->n);
                MatNd_softMax(a, action, action->m);  // action->m is a neat heuristic for beta
                x_dot += MatNd_get(a, i, 0)*x_dot_ds;

                MatNd_set(activation, i, 0, MatNd_get(a, i, 0));
                MatNd_destroy(a);
                break;
            }

            case TaskCombinationMethod::Product:
            {
                // Create temp matrix
                MatNd* otherActions = NULL; // other actions are all actions without the current
                MatNd_clone2(otherActions, action);
                MatNd_deleteRow(otherActions, i);

                // Treat the actions as probability of events and compute the probability that all other actions are false
                double prod = 1;  // 1 is the neutral element for multiplication
                for (unsigned int a = 0; a < otherActions->m; a++)
                {
                    REXEC(7)
                    {
                        std::cout << "factor " << (1 - otherActions->ele[a]) << std::endl;
                    }
                    // One part of the product is always 1
                    prod *= (1 - otherActions->ele[a]);
                }
                REXEC(7)
                {
                    std::cout << "prod " << prod << std::endl;
                }

                x_dot += action->ele[i]*prod*x_dot_ds;
                MatNd_set(activation, i, 0, action->ele[i]*prod);

                MatNd_destroy(otherActions);
                break;
            }
        }

        // Print if debug level is exceeded
        REXEC(5)
        {
            std::cout << "action DS " << i << " = " << action->ele[i] << std::endl;
            std::cout << "x_dot DS " << i << " =" << std::endl << x_dot_ds << std::endl;
        }
    }

    if (taskCombinationMethod == TaskCombinationMethod::Mean)
    {

        double normalizer = MatNd_getNormL1(action) + 1e-8;
        x_dot /= normalizer;
        MatNd_constMulSelf(activation, 1./normalizer);
    }

    // Integrate to x
    x += x_dot*dt;

    // Pass x to wrapped action model
    MatNd x_rcs = viewEigen2MatNd(x);

    // Compute the joint angle positions (joint angle velocities, and torques)
    wrapped->computeCommand(q_des, q_dot_des, T_des, &x_rcs, dt);

    // Print if debug level is exceeded
    REXEC(5)
    {
        std::cout << "x_dot (combined MP) =\n" << x_dot << std::endl;
        std::cout << "x (combined MP) =\n" << x << std::endl;
    }
    REXEC(7)
    {
        MatNd_printComment("q_des", q_des);
        MatNd_printComment("q_dot_des", q_dot_des);
        MatNd_printComment("T_des", T_des);
    }
}

void AMTaskActivation::reset()
{
    wrapped->reset();
    // Initialize shapes
    x.setZero(wrapped->getDim());
    x_dot.setZero(wrapped->getDim());

    // Obtain current stable action from wrapped action model
    MatNd x_rcs = viewEigen2MatNd(x);
    wrapped->getStableAction(&x_rcs);
}

void AMTaskActivation::getStableAction(MatNd* action) const
{
    // All zero activations is stable
    MatNd_setZero(action);
}

Eigen::VectorXd AMTaskActivation::getX() const
{
    return x;
}

Eigen::VectorXd AMTaskActivation::getXdot() const
{
    return x_dot;
}

ActionModel* AMTaskActivation::getWrappedActionModel() const
{
    return wrapped;
}

const std::vector<DynamicalSystem*>& AMTaskActivation::getDynamicalSystems() const
{
    return dynamicalSystems;
}

ActionModel* AMTaskActivation::clone(RcsGraph* newGraph) const
{
    std::vector<DynamicalSystem*> dsvc;
    for (auto ds : dynamicalSystems)
    {
        dsvc.push_back(ds->clone());
    }

    return new AMTaskActivation(wrapped->clone(newGraph), dsvc, TaskCombinationMethod::Mean);
}

TaskCombinationMethod AMTaskActivation::checkTaskCombinationMethod(std::string tcmName)
{
    TaskCombinationMethod tcm;
    if (tcmName == "sum")
    {
        tcm = TaskCombinationMethod::Sum;
    }
    else if (tcmName == "mean")
    {
        tcm = TaskCombinationMethod::Mean;
    }
    else if (tcmName == "softmax")
    {
        tcm = TaskCombinationMethod::SoftMax;
    }
    else if (tcmName == "product")
    {
        tcm = TaskCombinationMethod::Product;
    }
    else
    {
        std::ostringstream os;
        os << "Unsupported task combination method: " << tcmName;
        throw std::invalid_argument(os.str());
    }
    return tcm;
}

const char* AMTaskActivation::getTaskCombinationMethodName() const
{
    if (taskCombinationMethod == TaskCombinationMethod::Sum)
    {
        return "sum";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Mean)
    {
        return "mean";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::SoftMax)
    {
        return "softmax";
    }
    else if (taskCombinationMethod == TaskCombinationMethod::Product)
    {
        return "product";
    }
    else
    {
        return nullptr;
    }
}

MatNd* AMTaskActivation::getActivation() const
{
    return activation;
}

} /* namespace Rcs */

