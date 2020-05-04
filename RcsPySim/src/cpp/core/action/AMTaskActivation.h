#ifndef _AMTASKACTIVATION_H_
#define _AMTASKACTIVATION_H_

#include "ActionModel.h"
#include "DynamicalSystem.h"

namespace Rcs
{
    
    /*! Combination method for tasks a.k.a. movement primitives
     * Determines how the contribution of each task is scaled with its activation.
     */
    enum class TaskCombinationMethod
    {
        Sum, Mean, SoftMax, Product
    };
    
    /*! Action model controlling the activations of multiple tasks. Each task is defined by a DynamicalSystem.
     * For every task, there is one activation variable as part of the action space.
     * The activation is a value between 0 and 1, where 0 means to ignore the task
     * completely. The activation values do not need to sum to 1.
     */
    class AMTaskActivation : public ActionModel
    {
    public:
        
        /*! Constructor
         * @param[in] wrapped ActionModel defining the output space of the tasks (takes ownership)
         * @param[in] ds      List of dynamical systems (takes ownership)
         * @param[in] tcm     Mode that determines how the different tasks a.k.a. movement primitives are combined
         */
        explicit AMTaskActivation(ActionModel* wrapped, std::vector<DynamicalSystem*> ds, TaskCombinationMethod tcm);
        
        virtual ~AMTaskActivation();
        
        // not copy- or movable - klocwork doesn't pick up the inherited ones. RCSPYSIM_NOCOPY_NOMOVE(AMTaskActivation)
        
        virtual ActionModel* clone(RcsGraph* newGraph) const;
        
        //! Get the number of DS, i.e. entries in the dynamicalSystems vector, owned by the action model
        virtual unsigned int getDim() const;
        
        virtual void getMinMax(double* min, double* max) const;
        
        virtual std::vector<std::string> getNames() const;
        
        virtual void computeCommand(MatNd* q_des, MatNd* q_dot_des, MatNd* T_des, const MatNd* action, double dt);
        
        virtual void reset();
        
        virtual void getStableAction(MatNd* action) const;
        
        Eigen::VectorXd getX() const;
        
        Eigen::VectorXd getXdot() const;
        
        virtual ActionModel* getWrappedActionModel() const;
        
        //! Get a vector of the owned dynamical systems.
        const std::vector<DynamicalSystem*>& getDynamicalSystems() const;
        
        MatNd* getActivation() const;
        
        static TaskCombinationMethod checkTaskCombinationMethod(std::string tcmName);
        
        const char* getTaskCombinationMethodName() const;
    
    protected:
        //! wrapped action model
        ActionModel* wrapped;
        //! list of dynamical systems
        std::vector<DynamicalSystem*> dynamicalSystems;
        //! current state in task space
        Eigen::VectorXd x;
        //! current velocity in task space
        Eigen::VectorXd x_dot;
        //! the activation resulting from the action and the task combination method (used for logging)
        MatNd* activation;
        //! way to combine the tasks' contribution
        TaskCombinationMethod taskCombinationMethod;
    };
    
} /* namespace Rcs */

#endif /* _AMTASKACTIVATION_H_ */
