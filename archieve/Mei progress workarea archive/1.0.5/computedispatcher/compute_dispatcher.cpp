#include "compute_dispatcher.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#elif defined(__has_include)
    #if __has_include(<CL/cl.h>)
        #include <CL/cl.h>
        #define HAS_OPENCL
    #endif
#endif

std::unique_ptr<ComputeDispatcher> CreateDispatcher(ComputeTarget& selected_target) {
#ifdef HAS_OPENCL
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "[AutoDetect] No OpenCL platforms found. Using CPU backend.\n";
        selected_target = ComputeTarget::CPU;
        return std::make_unique<CPUDispatcher>();
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (auto platform : platforms) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices > 0) {
            std::cerr << "[AutoDetect] GPU detected. Using GPU backend.\n";
            selected_target = ComputeTarget::GPU;
            return std::make_unique<GPUDispatcher>();
        }
    }

    std::cerr << "[AutoDetect] No usable GPU devices. Using CPU backend.\n";
    selected_target = ComputeTarget::CPU;
    return std::make_unique<CPUDispatcher>();
#else
    std::cerr << "[AutoDetect] OpenCL not available. Using CPU backend.\n";
    selected_target = ComputeTarget::CPU;
    return std::make_unique<CPUDispatcher>();
#endif
}
