#ifndef SRC_CPP_CORE_UTIL_NOCOPY_H_
#define SRC_CPP_CORE_UTIL_NOCOPY_H_


#define RCSPYSIM_NOCOPY(cls)\
    cls(const cls&) = delete;\
    cls& operator=(const cls&) = delete;
#define RCSPYSIM_NOMOVE(cls)\
    cls(cls&&) = delete;\
    cls& operator=(cls&&) = delete;
#define RCSPYSIM_NOCOPY_NOMOVE(cls)\
    RCSPYSIM_NOCOPY(cls)\
    RCSPYSIM_NOMOVE(cls)


#endif /* SRC_CPP_CORE_UTIL_NOCOPY_H_ */
