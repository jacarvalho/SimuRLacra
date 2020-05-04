#include "PhysicsSimulationComponent.h"
#include "PhysicsFactory.h"

#include <Rcs_joint.h>
#include <Rcs_macros.h>
#include <Rcs_typedef.h>
#include <Rcs_timer.h>



/*******************************************************************************
 * Constructor
 ******************************************************************************/
Rcs::PhysicsSimulationComponent::PhysicsSimulationComponent(RcsGraph* graphCurr,
                                                            const char* engine,
                                                            const char* cfgFile) :
  Rcs::HardwareComponent(),
  PeriodicCallback(),
  currentGraph(graphCurr),
  dt(0.0),
  dtSim(0.0),
  tStart(Timer_getSystemTime()),
  sim(NULL),
  mtx(NULL),
  ffwd(false)
{
  // setSynchronizationMode(Synchronize_Hard);
  setClassName(getName());
  this->sim = Rcs::PhysicsFactory::create(engine, graphCurr, cfgFile);
  RCHECK(this->sim);
  // sim->setMassAndInertiaFromPhysics(graph);
  // sim->setJointLimits(false);
  // sim->disableCollisions();
  // setSchedulingPolicy(SCHED_RR);
  setSynchronizationMode(Synchronize_Hard);
}

/*******************************************************************************
 * Constructor
 ******************************************************************************/
Rcs::PhysicsSimulationComponent::PhysicsSimulationComponent(PhysicsBase* sim) :
        Rcs::HardwareComponent(),
        PeriodicCallback(),
        currentGraph(const_cast<RcsGraph*>(sim->getGraph())),
        dt(0.0),
        dtSim(0.0),
        tStart(Timer_getSystemTime()),
        sim(sim),
        mtx(NULL),
        ffwd(false)
{
    RCHECK(sim);
    setClassName(getName());
    setSynchronizationMode(Synchronize_Hard);
}

/*******************************************************************************
 * Destructor
 ******************************************************************************/
Rcs::PhysicsSimulationComponent::~PhysicsSimulationComponent()
{
  stop();
  delete this->sim;
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::start(double updateFreq, int prio)
{
  this->dt = 1.0/updateFreq;
  this->tStart = Timer_getSystemTime();
  PeriodicCallback::start(updateFreq, prio);
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::setMutex(pthread_mutex_t* mtx)
{
  this->mtx = mtx;
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::lock()
{
  if (this->mtx != NULL)
  {
    pthread_mutex_lock(this->mtx);
  }
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::unlock()
{
  if (this->mtx != NULL)
  {
    pthread_mutex_unlock(this->mtx);
  }
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::updateGraph(RcsGraph* graph)
{
  double tmp = Timer_getSystemTime();

  lock();

  if (this->ffwd==true)
  {
    sim->getLastPositionCommand(graph->q);
  }
  else
  {
    if (this->dt>0.0)
    {
      sim->simulate(this->dt, graph, NULL, NULL, true);
    }
  }

  unlock();

  this->dtSim = Timer_getSystemTime() - tmp;
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::setCommand(const MatNd* q_des,
                                                 const MatNd* q_dot_des,
                                                 const MatNd* T_des)
{
  sim->setControlInput(q_des, q_dot_des, T_des);
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::tare()
{
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::setEnablePPS(bool enable)
{
  sim->setEnablePPS(enable);
}

/*******************************************************************************
 *
 ******************************************************************************/
const char* Rcs::PhysicsSimulationComponent::getName() const
{
  return "PhysicsSimulation";
}

/*******************************************************************************
 *
 ******************************************************************************/
double Rcs::PhysicsSimulationComponent::getCallbackUpdatePeriod() const
{
  return this->dt;
}

/*******************************************************************************
 *
 ******************************************************************************/
double Rcs::PhysicsSimulationComponent::getLastUpdateTime() const
{
  return 0.0;// TODO
}

/*******************************************************************************
 * Thread callback, periodically called.
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::callback()
{
  callControlLayerCallbackIfConnected();
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::getLastPositionCommand(MatNd* q_des) const
{
  sim->getLastPositionCommand(q_des);
}

/*******************************************************************************
 *
 ******************************************************************************/
Rcs::PhysicsBase* Rcs::PhysicsSimulationComponent::getPhysicsSimulation() const
{
  return this->sim;
}

/*******************************************************************************
 *
 ******************************************************************************/
int Rcs::PhysicsSimulationComponent::sprint(char* str, size_t size) const
{
  return snprintf(str, size, "Simulation time: %.3f (%.3f)\nStep took %.1f msec\n",
                  sim->time(), Timer_getSystemTime()-this->tStart, dtSim*1.0e3);
}

/*******************************************************************************
 *
 ******************************************************************************/
void Rcs::PhysicsSimulationComponent::setFeedForward(bool ffwd_)
{
  this->ffwd = ffwd_;
}

/*******************************************************************************
 *
 ******************************************************************************/
double Rcs::PhysicsSimulationComponent::getStartTime() const
{
  return this->tStart;
}

bool Rcs::PhysicsSimulationComponent::startThread()
{
  start(getUpdateFrequency(), getThreadPriority());
  return true;
}

bool Rcs::PhysicsSimulationComponent::stopThread()
{
  stop();
  return true;
}
