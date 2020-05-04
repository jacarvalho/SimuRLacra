#ifndef _CONTROLPOLICY_H_
#define _CONTROLPOLICY_H_

#include <Rcs_MatNd.h>

#include <vector>
#include <string>

namespace Rcs
{

class PropertySource;

/**
 * Base class for a control policy that computes the actions from a given observation vector.
 */
class ControlPolicy
{
public:
    // static registry

    //! Policy factory function. Should read a policy from a file.
    typedef ControlPolicy* (*ControlPolicyCreateFunction)(const char*);
  
  /*! Register a control policy type.
   * @param[in] name policy type name
   * @param[in] creator factory function
   */
  static void registerType(const char* name, ControlPolicyCreateFunction creator);

    /*! Load a saved policy.
     * @param[in] name policy type name
     * @param[in] dataFile file to load
     * @return loaded policy
     */
    static ControlPolicy* create(const char* name, const char* dataFile);

    /*! Load a saved policy defined by the given configuration.
     * @param[in] config property config, containing type and file entries
     * @return loaded policy
     */
    static ControlPolicy* create(PropertySource* config);

    //! List available policy names.
    static std::vector<std::string> getTypeNames();

    ControlPolicy();
    virtual ~ControlPolicy();

    /*! Reset internal state if any.
     * The default implementation does nothing.
     */
    virtual void reset();

    /*!
     * Compute the action according to the policy.
     * @param[out] action matrix to store the action in
     * @param[in]  observation current observed state
     */
    virtual void computeAction(MatNd* action, const MatNd* observation) = 0;
};

/**
 * Create a static field of this type to register a control policy type.
 */
template<class Policy>
class ControlPolicyRegistration
{
public:
    /**
     * Register the template type under the given name.
     * @param name experiment name
     */
    explicit ControlPolicyRegistration(const char* name)
    {
        ControlPolicy::registerType(name, ControlPolicyRegistration::create);
    }

    /**
     * Creator function passed to ExperimentConfig::registerType.
     */
    static ControlPolicy* create(const char* dataFile)
    {
        return new Policy(dataFile);
    }

};

} /* namespace Rcs */

#endif /* _CONTROLPOLICY_H_ */
