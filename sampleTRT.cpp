#include <iostream>
#include <string>
#include <cstdlib>
#ifndef _WIN32
#include <dlfcn.h>
std::string libname = "./libsampleTRTLib.so";
#else
#include <windows.h>
#define dlopen(name, flags) LoadLibrary(name)
#define dlsym(handle,name) GetProcAddress((HMODULE)handle, name)
std::string libname = "sampleTRTLib.dll";
#endif

typedef int (*tensorRTRunTest)(const char*, const char*);

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    const char* modfolder = "data/model";
    const char* infile = "data/input/input.rgb";

    if ( argc > 1 ) {
        modfolder = argv[1];
    }
    if ( argc > 2 ) {
        infile = argv[2];
    }


    std::cout << "Loading library from " << libname << std::endl;
    void* lib = dlopen(libname.c_str(), RTLD_LAZY|RTLD_GLOBAL);
    if ( !lib ) {
        std::cout << "Failed to load " << libname << std::endl;
        return -1;
    }
    tensorRTRunTest testFun = (tensorRTRunTest)dlsym(lib, "tensorRTRunTest");
    if ( !testFun ) {
        std::cout << "Failed to load tensorRTRunTest from " << libname << std::endl;
        return -1;
    }
    return testFun(modfolder, infile);
}
