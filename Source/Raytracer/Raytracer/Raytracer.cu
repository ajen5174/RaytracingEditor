#include "Raytracer.h"
#include "Math/Sphere.h"
#include "Math/Triangle.h"
#include "Renderer/Mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

__global__ void FreeWorld(Triangle** triangles, int numVertices, vec3* vertices, Camera** cam)
{
    for (int i = 0; i < numVertices; i++)
    {
        delete triangles[i];
    }

    delete* cam;
}
__global__ void CreateWorld(Triangle** triangles, int numVertices, vec3* vertices, Camera** cam, int width, int height)
{
    *cam = new Camera(vec3(0.0f), vec3(0.0f), vec3(0.0f), 45.0f, (float)width / (float)height);
    //*object = new Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
    //*object = new Triangle(vec3(-0.5f, -0.5f, -1.0f), vec3(0.5f, -0.5f, -1.0f), vec3(0.0f, 0.5f, -1.0f));
    //*object2 = new Triangle(vec3(-0.5f, -0.5f, -1.0f), vec3(-1.75f, 0.0f, -3.0f), vec3(0.0f, 0.5f, -1.0f));
    

    for (int i = 0; i < numVertices; i++)
    {
        if (i % 3 == 0)
        {
            triangles[i / 3] = new Triangle(vertices[i + 0], vertices[i + 1], vertices[i + 2]);
        }


        //int triIndex = i / 3;
        //triangles[triIndex]->vertices[i % 3] = vertices[i];
        //int x;
    }

    //transform the triangles lmao
    for (int i = 0; i < numVertices / 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            triangles[i]->vertices[j].z -= 4.0f;
        }
    }


}


__device__ vec3 GetColor(Triangle** triangles, int numTriangles, const Ray& r)
{
    HitInfo info;
    float closestSoFar = 100.0f;
    for (int i = 0; i < numTriangles; i++)
    {
        if (triangles[i]->Hit(r, 0.0f, closestSoFar, info))
        {
            closestSoFar = info.distance;
        }
    }

    if (info.u >= 0.0f)
    {
        return (info.normal + vec3(1.0f)) / 2.0f;
        return vec3(1.0f * info.u, 1.0f * info.v, 1.0f * info.w);
    }
    
    vec3 unitDir = Normalize(r.direction);
    float t = 0.5f * (unitDir.y + 1.0f);//based on how high it is, change the weight of the color from white to light blue
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void Render(vec3* frameBuffer, int width, int height, Camera** cam, Triangle** triangles, int numTriangles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    float u = (float)i / (float)width;
    float v = (float)j / (float)height;
    frameBuffer[pixel_index] = GetColor(triangles, numTriangles, (*cam)->GetRay(u, v));

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

    
    Camera** cam;
    CheckCudaErrors(cudaMalloc((void**)&cam, sizeof(Camera*)));

    //model loading?

    //Mesh** mesh;
    //CheckCudaErrors(cudaMalloc((void**)&mesh, sizeof(Mesh*)));


    std::string modelPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Meshes\\suzanne.obj");
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);
    
    int numVertices = scene->mMeshes[0]->mNumVertices;
    //vec3* vertices = new vec3[numVertices];
    vec3* vertices;// = new Triangle[numVertices / 3];
    CheckCudaErrors(cudaMallocManaged(&vertices, (numVertices) * sizeof(vec3)));

    for (int i = 0; i < scene->mMeshes[0]->mNumVertices; i++)
    {
        aiVector3D temp = scene->mMeshes[0]->mVertices[i];
        vertices[i].x = temp.x;
        vertices[i].y = temp.y;
        vertices[i].z = temp.z;
    }

    //float* floatVertices = new float[numVertices * 3];



    Triangle** triangles;// = new Triangle[numVertices / 3];
    CheckCudaErrors(cudaMallocManaged(&triangles, (numVertices / 3) * sizeof(Triangle*)));


    


    //Hittable** object;
    //CheckCudaErrors(cudaMalloc((void**)&object, sizeof(Hittable*)));
    //
    //Hittable** object2;
    //CheckCudaErrors(cudaMalloc((void**)&object2, sizeof(Hittable*)));

    CreateWorld<<<1, 1>>>(triangles, numVertices, vertices, cam, width, height);

    dim3 blocks(width / threadX + 1, height / threadY + 1);
    dim3 threads(threadX, threadY);

    Render <<<blocks, threads>>> (frameBuffer, width, height, cam, triangles, numVertices / 3);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << seconds << " seconds.\n";

    FreeWorld << <1, 1 >> > (triangles, numVertices, vertices, cam);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    //CheckCudaErrors(cudaFree(triangles));
    CheckCudaErrors(cudaFree(vertices));
    //CheckCudaErrors(cudaFree(cam));
    

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
