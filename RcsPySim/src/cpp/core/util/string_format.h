#ifndef _STRING_FORMAT_H_
#define _STRING_FORMAT_H_

#include "BoxSpace.h"

#include <string>
#include <cstdio>
#include <memory>

namespace Rcs
{

// Taken from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf, licensed under CC0 1.0
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // we don't want the '\0' inside
}

}

#endif //_STRING_FORMAT_H_
