
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "Math/vec3.h"
#include "Math/Ray.h"
#include "Raytracer.h"


int main()
{
    Raytracer* rt = new Raytracer("", "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Raytracer\\Outputs\\output.ppm");

    
    rt->StartRender();
    rt->WriteToFile();

    std::cout << "File written!";
    //CheckCudaErrors(cudaFree(frameBuffer));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

