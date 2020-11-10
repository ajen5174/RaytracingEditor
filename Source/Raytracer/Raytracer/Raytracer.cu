#include "Raytracer.h"
#include "Math/Sphere.h"
#include "Math/Triangle.h"
//#include "Renderer/Mesh.h"
//#include "Core/Entity.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <curand_kernel.h>
#include "Core/Json.h"


//
//__global__ void FreeWorld(Camera** cam)
//{
//    //for (int i = 0; i < mesh->numTriangles; i++)
//    //{
//    //    delete mesh->triangles[i];
//    //}
//
//    delete* cam;
//}

__global__ void CreateEntity(Entity* entity)
{
    entity = new Entity();

}


__device__ bool HitWorld(Entity** list, int count, const Ray& r, HitInfo& info)
{
    float closestSoFar = 100.0f;
    bool hit = false;
    HitInfo tempInfo;
    for (int j = 0; j < count; j++)
    {
        if (list[j]->mesh)
        {
            //if (list[j]->mesh->boundingSphere->Hit(r, 0.0f, 100.0f, info)) //bug here for suzanne
            {
                for (int i = 0; i < list[j]->mesh->numTriangles; i++)
                {
                    if (list[j]->mesh->triangles[i]->Hit(r, 0.001f, closestSoFar, tempInfo))
                    {
                        info = tempInfo;
                        closestSoFar = info.distance;
                        hit = true;

                    }
                }
            }
        }
    }
    return hit;
}


__device__ vec3 GetColor(Entity** list, int count,  const Ray& r, int maxRecursion, curandState* localRandState)
{
    Ray currentRay = r;
    vec3 currentAttenuation = vec3(1.0f);
    float tempAtt = 1.0f;

    for (int k = 0; k < maxRecursion; k++)
    {
        HitInfo info;
        if (HitWorld(list, count, currentRay, info))
        {
            Ray scattered;
            vec3 attenuation;
             //Material* l = info.material;
             //return info.material->albedo;

            //vec3 target = info.point + info.normal + RandomInUnitSphere(localRandState);
            //tempAtt *= 0.5f;
            //currentRay = Ray(info.point, target - info.point);

            if (info.material->Scatter(currentRay, info, attenuation, scattered, localRandState))
            {
                currentAttenuation = currentAttenuation * attenuation; //color multiplication
                currentRay = scattered;
                tempAtt *= 0.5f;
            }
            else
            {
                return vec3(0.0f);
            }
        }
        else
        {
            int temp = k;
            vec3 unitDir = Normalize(currentRay.direction);
            float t = 0.5f * (unitDir.y + 1.0f);//based on how high it is, change the weight of the color from white to light blue
            vec3 backgroundColor = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return currentAttenuation * backgroundColor;
        }
    }
    

    
    
    return vec3(0.0f); // recursion exceeded
    
}

__global__ void RenderInit(int width, int height, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curand_init(1984, pixel_index, 0, &randState[pixel_index]);
}

__global__ void Render(vec3* frameBuffer, int width, int height, int samples, int maxRecursion, Camera* cam, Entity** list, int numEntities, curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curandState localRandState = randState[pixel_index];

    

    vec3 color(0.0f);
    for (int k = 0; k < samples; k++)//number of samples
    {
        float u = (i + curand_uniform(&localRandState)) / (float)width;
        float v = (j + curand_uniform(&localRandState)) / (float)height;
        Ray r = cam->GetRay(u, v);
        color = color + GetColor(list, numEntities, r, maxRecursion, &localRandState);
    }

    color = color / (float)samples;
    color.x = sqrt(color.x);//sqrt gives a more accurate color, this is gamma correction
    color.y = sqrt(color.y);
    color.z = sqrt(color.z);

    frameBuffer[pixel_index] = color;

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
    std::cout << "Loading scene...\n";
    rapidjson::Document doc;
    if (json::LoadFromFile(sceneToLoad, doc))
    {
        
        json::GetInt(doc, "EntityCount", numEntities);

        CheckCudaErrors(cudaMallocManaged(&entityList, sizeof(Entity*) * numEntities));

        rapidjson::Value& enititiesArray = doc["entities"];
        if (enititiesArray.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < enititiesArray.Size(); i++)
            {
                rapidjson::Value& entityValue = enititiesArray[i];
                if (entityValue.IsObject())
                {
                    Entity* entity;
                    CheckCudaErrors(cudaMallocManaged(&entity, sizeof(Entity)));
                    CreateEntity<<<1, 1>>>(entity);
                    CheckCudaErrors(cudaDeviceSynchronize());
                    //entity->scene = this;
                    if (entity->Load(entityValue))
                    {
                        //add the entity to a list somewhere?
                        //entity;
                        entityList[i] = entity;
                    }
                    else
                    {
                        delete entity;
                    }
                }
            }
        }
    }

    for (int i = 0; i < numEntities; i++)
    {
        if (entityList[i]->cam)
        {
            if (entityList[i]->cam->isMainCam)
            {
                this->mainCamera = entityList[i]->cam;
            }
        }
    }

    std::cout << "Scene Loaded!\n";
    return true;
}



Raytracer::Raytracer(char** args)
    :renderPath(args[2])
{
    //std::cout << "Argument settings used...\n";
    LoadScene(args[1]);
    samplesPerPixel = std::stoi(args[3]);
    maxRecursion = std::stoi(args[4]);
    width = std::stoi(args[5]);
    height = std::stoi(args[6]);
}

Raytracer::Raytracer(std::string sceneToLoad, std::string renderPath)
    :renderPath(renderPath)
{
    //std::cout << "Default settings used...\n";
    LoadScene(sceneToLoad);
    samplesPerPixel = 10;
    maxRecursion = 50;
    width = 266.66666666f;
    height = 150.0f;
}

bool Raytracer::StartRender()
{
    std::cout << "Initializing Render...\n";
    
    int numPixels = width * height;

    curandState* randState;
    CheckCudaErrors(cudaMallocManaged(&randState, numPixels * sizeof(curandState)));

    size_t frameBufferSize = numPixels * sizeof(vec3);
    CheckCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    int threadX = 8;
    int threadY = 8;

    clock_t start, stop;
    start = clock();
    

    dim3 blocks(width / threadX + 1, height / threadY + 1);
    dim3 threads(threadX, threadY);


    RenderInit << <blocks, threads >> > (width, height, randState);
    CheckCudaErrors(cudaDeviceSynchronize());
    std::cout << "Initialized!\n";
    std::cout << "Rendering...\n";
    Render <<<blocks, threads>>> (frameBuffer, width, height, samplesPerPixel, maxRecursion, mainCamera, entityList, numEntities, randState);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Render took " << seconds << " seconds.\n";

    //FreeWorld << <1, 1 >> > (cam);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());


    //CheckCudaErrors(cudaFree(mesh));
    

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
