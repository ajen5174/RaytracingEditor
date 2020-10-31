

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "Raytracer.h"


int main(int argc, char** argv)
{
    Raytracer* rt;
    if (argc == 3)
    {
        rt = new Raytracer(argv[1], argv[2]);
    }
    else 
    {
        rt = new Raytracer("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Scenes\\scene.txt", 
                                  "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Raytracer\\Outputs\\output.ppm");

    }

    
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

