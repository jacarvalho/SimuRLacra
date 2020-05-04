#include "ExperimentConfig.h"
#include "action/AMJointControlPosition.h"
#include "action/AMJointControlAcceleration.h"
#include "action/AMIntegrate1stOrder.h"
#include "action/AMIntegrate2ndOrder.h"
#include "initState/ISSQuanserQube.h"
#include "observation/OMBodyStateLinear.h"
#include "observation/OMBodyStateAngular.h"
#include "observation/OMCombined.h"
#include "observation/OMJointState.h"
#include "physics/PhysicsParameterManager.h"
#include "physics/PPDMassProperties.h"
#include "physics/PPDMaterialProperties.h"
#include "physics/PPDRodLength.h"
#include "physics/ForceDisturber.h"
#include "util/string_format.h"

#include <Rcs_typedef.h>
#include <Rcs_macros.h>

#ifdef GRAPHICS_AVAILABLE
#include <RcsViewer.h>
#endif

#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>


namespace Rcs
{

class ECQuanserQube: public ExperimentConfig
{

protected:
    virtual ActionModel* createActionModel()
    {
        std::string actionModelType = "joint_acc";
        properties->getProperty(actionModelType, "actionModelType");

        if (actionModelType == "joint_acc")
        {
            /** V_m_max = 5 V and theta_dot_0 = 0 rad/s
             *  tau_max = k_m (V_m_max - k_m * theta_dot_0) / R_m = 0.04375 Nm
             *  J = m * l^2 / 12 = 0.000033282 kg m^2
             *  alpha_ddot_max = tau_max / J = 1314.52 rad/s^2 */
            double max_action = 1314.52 / 10.; // divide by 10 since if seem unrealistic otherwise
            properties->getProperty(max_action, "maxAction");
            // By integrating the position command twice before applying it as action we command the acceleration
            return new AMIntegrate2ndOrder(new AMJointControlPosition(graph), max_action);
            // return new AMJointControlAcceleration(graph); // not working
        }
        else
        {
          std::ostringstream os;
          os << "Unsupported action model type: " << actionModelType;
          throw std::invalid_argument(os.str());
        }
    }

    virtual ObservationModel* createObservationModel()
    {
        auto fullState = new OMCombined();
        fullState->addPart(OMJointState::observeAllJoints(graph));
        return fullState;
    }

    virtual void populatePhysicsParameters(PhysicsParameterManager* manager)
    {
        manager->addParam("Arm", new PPDRodLength());
        manager->addParam("Arm", new PPDMassProperties());
        manager->addParam("Arm", new PPDMaterialProperties());
        manager->addParam("Pendulum", new PPDRodLength());
        manager->addParam("Pendulum", new PPDMassProperties());
        manager->addParam("Pendulum", new PPDMaterialProperties());
    }

public:
    virtual InitStateSetter* createInitStateSetter()
    {
        return new ISSQuanserQube(graph);
    }

    virtual ForceDisturber* createForceDisturber()
    {
      RcsBody* arm = RcsGraph_getBodyByName(graph, "Arm");
      RcsBody* pendulum = RcsGraph_getBodyByName(graph, "Pendulum");
      return new ForceDisturber(arm, pendulum);
    }

    virtual void initViewer(Rcs::Viewer* viewer) {
#ifdef GRAPHICS_AVAILABLE
        viewer->setCameraHomePosition(osg::Vec3d(0.7, 0.7, 0.4),
                                      osg::Vec3d(-0.2, -0.2, 0.0),
                                      osg::Vec3d(0.0, 0.05, 1.0));
#endif
    }

    void getHUDText(std::vector<std::string>& linesOut, double currentTime, const MatNd *obs, const MatNd *currentAction,
                    PhysicsBase *simulator, PhysicsParameterManager *physicsManager, ForceDisturber* forceDisturber) override
    {
        // Obtain simulator name
        const char* simName = "None";
        if (simulator != NULL) {
            simName = simulator->getClassName();
        }

        // common stuff
        linesOut.emplace_back(
            string_format("physics engine: %s                 sim time:          %2.3f s",
                simName, currentTime));

        linesOut.emplace_back(
            string_format("                                     pole angles: [% 3.2f,% 3.2f] deg",
                RCS_RAD2DEG(obs->ele[0]), RCS_RAD2DEG(obs->ele[1])));

        const double* distForce = forceDisturber->getLastForce();
        linesOut.emplace_back(
            string_format("disturbances:   [% 3.1f,% 3.1f,% 3.1f] N",
                distForce[0], distForce[1], distForce[2]));

        std::stringstream ss;
        ss << "actions:      [";
        for (unsigned int i = 0; i < currentAction->m - 1; i++)
        {
            ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, i, 0) << ", ";
            if (i == 6)
            {
                ss << "\n               ";
            }
        }
        ss << std::fixed << std::setprecision(3) << MatNd_get(currentAction, currentAction->m - 1, 0) << "]";
        linesOut.emplace_back(string_format(ss.str()));

        if (physicsManager != NULL) {
            // Include physics parameters

            // Get bodies from graph
            RcsBody* arm = RcsGraph_getBodyByName(graph, "Arm");
            RcsBody* pendulum = RcsGraph_getBodyByName(graph, "Pendulum");


            linesOut.emplace_back(string_format(
                     "arm mass:       %1.3f kg          pendulum mass:           %1.3f kg",
                     arm->m, pendulum->m));

            linesOut.emplace_back(string_format(
                     "arm length:     %1.3f m         pendulum length:           %1.3f m",
                     arm->shape[0]->extents[2], pendulum->shape[0]->extents[2]));

            linesOut.emplace_back(string_format(
                     "arm radius:     %1.3f m         pendulum radius:           %1.3f m",
                     arm->shape[0]->extents[0],pendulum->shape[0]->extents[0]));
        }
    }
};

// register
static ExperimentConfigRegistration<ECQuanserQube> RegQuanserQube("QuanserQube");

}
