#include "Raytracer.h"
#include "Math/Sphere.h"
#include "Math/Triangle.h"
#include "Renderer/Mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

__global__ void FreeWorld(Mesh* mesh, Camera** cam)
{
    for (int i = 0; i < mesh->numTriangles; i++)
    {
        delete mesh->triangles[i];
    }

    delete* cam;
}
__global__ void CreateWorld(Mesh* mesh, Camera** cam, int width, int height)
{
    *cam = new Camera(vec3(0.0f), vec3(0.0f), vec3(0.0f), 45.0f, (float)width / (float)height);
    
    mesh = new Mesh();
    

}

//This method uses data read in from disk and stored in vertices to create our device only Triangles for collision. W
    //We need this method because the Triangle MUST be "new'd" on the DEVICE, or the vtable (hittable) won't be accessible by the device later.
__global__ void CreateMeshTriangles(Mesh* mesh)
{

    //transform the vertices lmao
    for (int i = 0; i < mesh->numTriangles * 3; i++)
    {
        mesh->vertices[i].z -= 6.0f;
    }

    for (int i = 0; i < mesh->numTriangles; i++)
    {
        //if (i % 3 == 0)
        {
            mesh->triangles[i] = new Triangle(mesh->vertices[(i * 3) + 0], mesh->vertices[(i * 3) + 1], mesh->vertices[(i * 3) + 2]);
        }


    }

    

    vec3 averagePosition;
    averagePosition = vec3();
    for (int i = 0; i < mesh->numTriangles * 3; i++)
    {
        averagePosition = averagePosition + mesh->vertices[i];
    }
    averagePosition = averagePosition / (mesh->numTriangles  *3);
    float boundingRadius = 0.0f;//arbitrarily large number, should be float max, but Im not sure what the macro for that is.
    for (int i = 0; i < mesh->numTriangles * 3; i++)
    {
        float tempRadius = (averagePosition - mesh->vertices[i]).Magnitude();
        if (tempRadius > boundingRadius)
            boundingRadius = tempRadius;
    }
    mesh->boundingSphere = new Sphere(averagePosition, boundingRadius);

}



__device__ vec3 GetColor(Mesh* mesh, const Ray& r)
{
    HitInfo info;
    float closestSoFar = 100.0f;
    if (mesh->boundingSphere->Hit(r, 0.0f, 100.0f, info))
    {
        for (int i = 0; i < mesh->numTriangles; i++)
        {
            if (mesh->triangles[i]->Hit(r, 0.0f, closestSoFar, info))
            {
                closestSoFar = info.distance;
            }
        }

        if (info.u >= 0.0f)
        {
            return (info.normal + vec3(1.0f)) / 2.0f;
            return vec3(1.0f * info.u, 1.0f * info.v, 1.0f * info.w);
        }
    }
    
    
    vec3 unitDir = Normalize(r.direction);
    float t = 0.5f * (unitDir.y + 1.0f);//based on how high it is, change the weight of the color from white to light blue
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void Render(vec3* frameBuffer, int width, int height, Camera** cam, Mesh* mesh) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    float u = (float)i / (float)width;
    float v = (float)j / (float)height;
    frameBuffer[pixel_index] = GetColor(mesh, (*cam)->GetRay(u, v));

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
    size_t frameBufferSize = numPixels * sizeof(vec3);

    CheckCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    int threadX = 8;
    int threadY = 8;

    clock_t start, stop;
    start = clock();

    //Allocate GPU memory blocks
    Camera** cam;
    CheckCudaErrors(cudaMalloc((void**)&cam, sizeof(Camera*)));

    Mesh* mesh;
    CheckCudaErrors(cudaMallocManaged(&mesh, sizeof(Mesh))); 

    CreateWorld<<<1, 1>>>(mesh, cam, width, height); //"new up" GPU memory (necessary for vtable access on GPU)
    CheckCudaErrors(cudaDeviceSynchronize());

    //read in memory
    std::string modelPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Meshes\\teapot.obj");
    CreateMesh(modelPath, mesh);//assign CPU/read-in memory to GPU allocated memory
    CreateMeshTriangles << <1, 1 >> > (mesh);//use CPU/read-in memory to create C++ objects on the GPU

    std::cout << "Mesh loaded.\n";


    dim3 blocks(width / threadX + 1, height / threadY + 1);
    dim3 threads(threadX, threadY);
    std::cout << "Rendering...\n";
    Render <<<blocks, threads>>> (frameBuffer, width, height, cam, mesh);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << seconds << " seconds.\n";

    FreeWorld << <1, 1 >> > (mesh, cam);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    //CheckCudaErrors(cudaFree(triangles));
    //CheckCudaErrors(cudaFree(vertices));
    //CheckCudaErrors(cudaFree(cam));
    CheckCudaErrors(cudaFree(mesh));
    

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
            size_t pixel_index = j * width + i;
            vec3 color = frameBuffer[pixel_index];
            int ir = int(255.99 * color.x);
            int ig = int(255.99 * color.y);
            int ib = int(255.99 * color.z);
            myFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    CheckCudaErrors(cudaFree(frameBuffer));
}
