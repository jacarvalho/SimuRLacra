#ifndef RCS_PHYSICSSIMULATIONCOMPONENT_H
#define RCS_PHYSICSSIMULATIONCOMPONENT_H


#include "HardwareComponent.h"

#include <PhysicsBase.h>
#include <PeriodicCallback.h>


namespace Rcs
{
class PhysicsSimulationComponent : public HardwareComponent, public PeriodicCallback
{
public:

  PhysicsSimulationComponent(RcsGraph* graph, const char* engine="Bullet",
                             const char* physicsCfgFile=NULL);
  PhysicsSimulationComponent(PhysicsBase* sim);
  virtual ~PhysicsSimulationComponent();

  virtual void updateGraph(RcsGraph* graph);
  virtual void setCommand(const MatNd* q_des, const MatNd* qp_des,
                          const MatNd* T_des);
  virtual void tare();
  virtual void setEnablePPS(bool enable);
  virtual const char* getName() const;
  virtual double getCallbackUpdatePeriod() const;
  virtual double getLastUpdateTime() const;
  virtual void getLastPositionCommand(MatNd* q_des) const;
  virtual void setFeedForward(bool ffwd);
  virtual PhysicsBase* getPhysicsSimulation() const;
  virtual int sprint(char* str, size_t size) const;
  virtual void start(double updateFreq=10.0, int prio=50);
  virtual void setMutex(pthread_mutex_t* mtx);
  virtual double getStartTime() const;

  bool startThread();
  bool stopThread();
private:
  virtual void callback();
  void lock();
  void unlock();

  RcsGraph* currentGraph;
  double dt, dtSim, tStart;
  PhysicsBase* sim;
  pthread_mutex_t* mtx;
  bool ffwd;
};

}

#endif   // RCS_PHYSICSSIMULATIONCOMPONENT_H
