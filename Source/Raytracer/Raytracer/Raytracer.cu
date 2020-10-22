#include "Raytracer.h"

__global__ void Render(float* frameBuffer, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width * 3 + i * 3;
    frameBuffer[pixel_index + 0] = float(i) / width;
    frameBuffer[pixel_index + 1] = float(j) / height;
    frameBuffer[pixel_index + 2] = 0.2;
}

void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
}
}

bool Raytracer::LoadScene(std::string sceneToLoad)
{

    return false;
}



Raytracer::Raytracer(std::string sceneToLoad, std::string renderPath)
    :renderPath(renderPath)
{
    LoadScene(sceneToLoad);
}

bool Raytracer::StartRender()
{

    width = 800;
    height = 600;
    int numPixels = width * height;
    size_t frameBufferSize = 3 * numPixels * sizeof(float);

    CheckCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    int threadX = 8;
    int threadY = 8;

    clock_t start, stop;
    start = clock();

    dim3 blocks(width / threadX + 1, height / threadY + 1);
    dim3 threads(threadX, threadY);
    Render << <blocks, threads >> > (frameBuffer, width, height);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << seconds << " seconds.\n";
    return false;
}

void Raytracer::WriteToFile()
{

    //not sure why but it won't use relative paths here?
    std::ofstream myFile(renderPath);

    // Output FB as Image
    myFile << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * 3 * width + i * 3;
            float r = frameBuffer[pixel_index + 0];
            float g = frameBuffer[pixel_index + 1];
            float b = frameBuffer[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            myFile << ir << " " << ig << " " << ib << "\n";
        }
    }
}
