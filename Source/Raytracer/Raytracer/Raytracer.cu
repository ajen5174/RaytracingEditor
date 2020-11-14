#include "Raytracer.h"
#include "Math/BVHNode.h"
#include "Math/AABB.h"
#include "Math/Sphere.h"
#include "Math/Triangle.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <curand_kernel.h>
#include "Core/Json.h"
#include "Renderer/World.h"


__global__ void CreateEntity(Entity* entity)
{
    entity = new Entity();

}


__device__ bool HitWorld(Entity** list, int count, float maxDist, const Ray& r, HitInfo& info)
{
    float closestSoFar = maxDist;
    bool hit = false;
    HitInfo tempInfo;
    for (int j = 0; j < count; j++)
    {
        if (list[j]->mesh)
        {
            //if (list[j]->mesh->triangles)
            //{
            //    if (list[j]->mesh->triangles[0]->Hit(r, 0.001f, closestSoFar, info))
            //    {
            //        return true;
            //    }
            //}
            
            if ((*list[j]->mesh->bvh))
            {
                if((*list[j]->mesh->bvh)->Hit(r, 0.001f, closestSoFar, tempInfo))
                {
                    info = tempInfo;
                    closestSoFar = info.distance;
                    hit = true;
                }
            }
            //if (list[j]->mesh->bvh->Hit(r, 0.001f, closestSoFar, tempInfo))
            //{
            //    info = tempInfo;
            //    closestSoFar = info.distance;
            //    hit = true;
            //}


            //if (list[j]->mesh->boundingSphere->Hit(r, 0.0f, 100.0f, info)) //bug here for suzanne
            //{
            //    for (int i = 0; i < list[j]->mesh->numTriangles; i++)
            //    {
            //        if (list[j]->mesh->triangles[i]->Hit(r, 0.001f, closestSoFar, tempInfo))
            //        {
            //            info = tempInfo;
            //            closestSoFar = info.distance;
            //            hit = true;

            //        }
            //    }
            //}
        }
    }
    return hit;
}
__device__ vec3 GetShadows(Entity** list, int numEntities, Light** lights, int numLights, const Ray& r, int maxRecursion, curandState* localRandState)
{

}


__device__ vec3 GetColor(Entity** list, int numEntities,  Light** lights, int numLights, const Ray& r, int maxRecursion, curandState* localRandState)
{
    Ray currentRay = r;
    vec3 currentAttenuation = vec3(0.0f);
    bool hitOnce = false;
    HitInfo info;
    for (int k = 0; k < maxRecursion; k++)
    {
        if (HitWorld(list, numEntities, 100.0f, currentRay, info))
        {
            Ray scattered;
            vec3 attenuation;
            hitOnce = true;
            if (info.material->Scatter(currentRay, info, attenuation, scattered, localRandState))
            {
                currentAttenuation = currentAttenuation + attenuation; //color multiplication
                currentRay = scattered;

            }
            else
            {
                return vec3(0.0f);
            }
        }
        else
        {
            //this seems like the time to use shadow rays for light calculations
                //grab current hit position
                    //if it didn't hit anything at all fill with background color?
            if (!hitOnce)
            {
                vec3 unitDir = Normalize(currentRay.direction);
                float t = 0.5f * (unitDir.y + 1.0f);//based on how high it is, change the weight of the color from white to light blue
                vec3 backgroundColor = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
                return backgroundColor;
            }
            //if it did hit something, and this is just the final bounce, then we cast a ray to each light to calculate whether we are in shadow.
            else
            {
                for (int i = 0; i < numLights; i++)
                {
                    vec3 pointToLight = lights[i]->owner->transform->translation - info.point;
                    float distance = pointToLight.Magnitude();
                    HitInfo shadowInfo;
                    //if we cast a ray, and hit nothing until the light, then the light affects this material
                     //so we multiply the attenuation by the light color instead of the background color
                    if (!HitWorld(list, numEntities, distance, Ray(info.point, pointToLight), shadowInfo))
                    {
                        float tempDot = Dot(Normalize(pointToLight), Normalize(info.normal));
                        float lDotN = tempDot > 0.0f ? tempDot : 0.0f;

                        if (tempDot < 0.0f)
                        {
                            return vec3(0.0f);
                        }
                        
                        currentAttenuation = currentAttenuation + (lights[i]->color * lights[i]->intensity) * lDotN;
                    }
                }
            }

            return currentAttenuation / (M_PI * k);
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

__global__ void Render(vec3* frameBuffer, int width, int height, int samples, int maxRecursion, Camera* cam, Entity** list, int numEntities, Light** lights, int numLights, curandState* randState) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curandState localRandState = randState[pixel_index];

    vec3 color(0.0f);
    float u = (i + curand_uniform(&localRandState)) / (float)width;
    float v = (j + curand_uniform(&localRandState)) / (float)height;
    Ray r = cam->GetRay(u, v);
    //color = GetShadows(list, numEntities, lights, numLights, r, maxRecursion, &localRandState);

    for (int k = 0; k < samples; k++)//number of samples
    {
        u = (i + curand_uniform(&localRandState)) / (float)width;
        v = (j + curand_uniform(&localRandState)) / (float)height;
        r = cam->GetRay(u, v);
        color = color + GetColor(list, numEntities, lights, numLights, r, maxRecursion, &localRandState);
    }

    color = color / (float)samples;
    //color.x = sqrt(color.x);//sqrt gives a more accurate color, this is gamma correction, this has been moved to the get color function when we divide by PI pretty sure
    //color.y = sqrt(color.y);
    //color.z = sqrt(color.z);

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

            

            for (int i = 0; i < numEntities; i++)
            {
                if (entityList[i]->light)
                    numLights++;
            }

            CheckCudaErrors(cudaMallocManaged(&lights, sizeof(Light*) * numLights));
            CheckCudaErrors(cudaDeviceSynchronize());

            int currentLightIndex = 0;
            for (int i = 0; i < numEntities; i++)
            {
                if (entityList[i]->light)
                {
                    if (currentLightIndex < numLights)
                    {
                        lights[currentLightIndex] = entityList[i]->light;
                        currentLightIndex++;
                    }
                    else
                    {
                        std::cout << "more lights than accounted for...";
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
    samplesPerPixel = 5;
    maxRecursion = 50;
    width = 266.6666666f;// 533.333333f;
    height = 150.0f;// 300.0f;
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
    Render <<<blocks, threads>>> (frameBuffer, width, height, samplesPerPixel, maxRecursion, mainCamera, entityList, numEntities, lights, numLights, randState);
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
