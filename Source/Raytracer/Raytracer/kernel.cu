

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "Raytracer.h"


int main(int argc, char** argv)
{
    //std::cout << "Argument count: " << std::to_string(argc) << '\n';

    //for (int i = 0; i < argc; i++)
    //{
    //    std::cout << "Argument " << i << ": " << argv[i] << '\n';
    //}

    cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 166535);

    size_t limit;
    cudaDeviceGetLimit(&limit, cudaLimit::cudaLimitStackSize);

    Raytracer* rt;
    if (argc == 7)
    {
        rt = new Raytracer(argv);
    }
    else 
    {
        //for debugging
        rt = new Raytracer("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Scenes\\suzanne_box.txt", 
                                  "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Raytracer\\Outputs\\output.ppm");
    }

    
    
    rt->StartRender();
    rt->WriteToFile();

    std::cout << "File written!\n";
    //CheckCudaErrors(cudaFree(frameBuffer));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    //std::cout << "Press enter to close...\n";
    system("pause");
    return 0;
}

