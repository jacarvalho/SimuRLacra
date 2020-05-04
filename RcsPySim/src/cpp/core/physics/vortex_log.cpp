#include "vortex_log.h"

#include <Rcs_macros.h>
#include <stdexcept>
#include <sstream>

#ifdef USE_VORTEX
// implement method
#include <Vx/VxMessage.h>

void Rcs::setVortexLogLevel(const char* levelStr)
{
    // convert string to enum value
    Vx::eLogLevel level;
    if (STRCASEEQ(levelStr, "off")) {
        level = Vx::kOff;
    } else if (STRCASEEQ(levelStr, "fatal")) {
        level = Vx::kFatal;
    } else if (STRCASEEQ(levelStr, "error")) {
        level = Vx::kError;
    } else if (STRCASEEQ(levelStr, "warn")) {
        level = Vx::kWarn;
    } else if (STRCASEEQ(levelStr, "info")) {
        level = Vx::kInfo;
    } else if (STRCASEEQ(levelStr, "debug")) {
        level = Vx::kDebug;
    } else if (STRCASEEQ(levelStr, "trace")) {
        level = Vx::kTrace;
    } else if (STRCASEEQ(levelStr, "all")) {
        level = Vx::kAll;
    } else {
        std::ostringstream os;
        os << "Unsupported vortex log level: " << levelStr;
        throw std::invalid_argument(os.str());
    }
    // set to vortex
    Vx::LogSetLevel(level);
}
#else
// vortex not available, show warning
void Rcs::setVortexLogLevel(const char* levelStr)
{
    RLOG(1, "Vortex physics engine is not supported.");
}
#endif



