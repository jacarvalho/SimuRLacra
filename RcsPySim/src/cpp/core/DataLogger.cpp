#include "DataLogger.h"

#include <Rcs_macros.h>
#include <Rcs_VecNd.h>

#include <sstream>

namespace Rcs
{


DataLogger::DataLogger(std::string fileBaseName) :
          baseFileName(fileBaseName),
          fileCounter(0),
          running(false),
          buffer(NULL),
          currentStep(0)
{
    // nothing else to do
}

DataLogger::~DataLogger() {
    // make sure we stopped
    stop();
}

void DataLogger::start(const BoxSpace* observationSpace,
        const BoxSpace* actionSpace, unsigned int maxStepCount,
        const char* filename)
{
    // guard against concurrency
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (running) {
        RLOG(1, "Already running!");
        return;
    }
    running = true;
    // determine filename
    std::string fname;
    if (filename == NULL) {
        // generate
        std::ostringstream os;
        os << baseFileName;
        os << (fileCounter++);
        os << ".csv";
        fname = os.str();
    } else {
        fname = filename;
    }
    // open output file
    output.open(fname);

    // write header (column names)
    output << R"("step","reward)";

    for (auto& name : observationSpace->getNames()) {
        output << "\",\"" << name;
    }
    for (auto& name : actionSpace->getNames()) {
        output << "\",\"" << name;
    }
    output << "\"" << std::endl;
    // allocate buffer
    buffer = MatNd_create(maxStepCount, 1 + observationSpace->getNames().size() + actionSpace->getNames().size());
    currentStep = 0;

    RLOG(0, "Logging started!");
}

void DataLogger::stop()
{
    // guard against concurrency
    std::unique_lock<std::recursive_mutex> lock(mutex);
    if (!running) {
        return;
    }
    running = false;

    // write buffer contents to csv
    for (unsigned int row = 0; row < currentStep; ++row) {

        // write step number
        output << row;

        // write elements
        for (unsigned int col = 0; col < buffer->n; ++col) {
            output << "," << MatNd_get2(buffer, row, col);
        }
        output << std::endl;
    }

    // close file
    output.flush();
    output.close();
    output.clear();
    // delete buffer
    MatNd_destroy(buffer);

    RLOG(0, "Logging stopped!");
}

void DataLogger::record(const MatNd* observation, const MatNd* action,
        double reward)
{
    // try to obtain lock. If it's blocked, it's blocked by start or stop, so we don't want to lock anyways
    std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock() || !running) {
        return;
    }
    // add a line to buffer
    if (currentStep >= buffer->m) {
        stop();
        return;
    }
    double* lineBuffer = MatNd_getRowPtr(buffer, currentStep++);

    lineBuffer[0] = reward;
    VecNd_copy(&lineBuffer[1], observation->ele, observation->m);
    VecNd_copy(&lineBuffer[1 + observation->m], action->ele, action->m);
}

} /* namespace Rcs */
