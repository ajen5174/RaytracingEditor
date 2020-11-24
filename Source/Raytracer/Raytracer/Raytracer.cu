#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "Raytracer.h"
#include "Math/BVHNode.h"
#include "Math/AABB.h"
#include "Math/Sphere.h"
#include "Math/Triangle.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <stb_image_write.h>
#include <shlwapi.h>
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
            if(list[j]->mesh->box.Hit(r, 0.1f, closestSoFar))
            {
                for (int i = 0; i < list[j]->mesh->numTriangles; i++)
                {
                    if (list[j]->mesh->triangles[i]->Hit(r, 0.1f, closestSoFar, tempInfo))
                    {
                        info = tempInfo;
                        closestSoFar = info.distance;
                        hit = true;

                    }
                }
            }

        }
        else if (list[j]->primitive)
        {
            if (list[j]->primitive->Hit(r, 0.1f, closestSoFar, tempInfo))
            {
                info = tempInfo;
                closestSoFar = info.distance;
                hit = true;
            }
        }
    }
    return hit;
}


__device__ vec3 GetColor(Entity** list, int numEntities, Light** lights, int numLights, const Ray& r, int maxRecursion, curandState* localRandState)
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
                currentAttenuation = currentAttenuation + attenuation; 
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
                    if (lights[i]->lightType == Light::LightType::POINT || lights[i]->lightType == Light::LightType::DIRECTION)
                    {
                        vec3 pointToLight = lights[i]->owner->transform->translation - info.point;
                        float distance = lights[i]->lightType == Light::LightType::POINT ? pointToLight.Magnitude() : 1000.0f;
                        HitInfo shadowInfo;
                        Ray shadowRay = lights[i]->lightType == Light::LightType::POINT ? Ray(info.point, Normalize(pointToLight)) : Ray(info.point, -Normalize(lights[i]->direction));
                        //if we cast a ray, and hit nothing until the light, then the light affects this material
                         //so we multiply the attenuation by the light color instead of the background color
                        vec3 unitDir = Normalize(currentRay.direction);
                        float t = 0.5f * (unitDir.y + 1.0f);//based on how high it is, change the weight of the color from white to light blue
                        vec3 backgroundColor = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
                        
                        if (info.material->materialType == 'm')
                        {
                            currentAttenuation = (currentAttenuation + ((backgroundColor) / M_PI));
                        }
                        else
                        {
                            currentAttenuation = (currentAttenuation + ((backgroundColor * 0.1f) / M_PI));
                        }
                        
                        if (!HitWorld(list, numEntities, distance, shadowRay, shadowInfo)) //super unsure why I need to do the -1.0f
                        {
                            float tempDot = Dot(Normalize(pointToLight), Normalize(info.normal));
                            float lDotN = tempDot > 0.0f ? tempDot : 0.0f;

                            if (tempDot == 0.0f) //negative number to prevent artifacts?
                            {
                                return vec3(0.0f);
                            }

                            currentAttenuation = currentAttenuation + (lights[i]->color * lights[i]->intensity * 2.5f) * lDotN;// / (distance * distance);
                           
                            
                        }
                    }
                    
                    
                }
            }
            
            return currentAttenuation / (M_PI * (k));
        }
    }
    

    
    return vec3(0.0f);
}

__global__ void RenderInit(int width, int height, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    curand_init(1984, pixel_index, 0, &randState[pixel_index]);
}

__global__ void Render(vec3* frameBuffer, int width, int height, int samples, int maxRecursion, Camera* cam, Entity** list, int numEntities, Light** lights, int numLights, volatile int* progress, curandState* randState) 
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
    int temp = *progress;
    temp += 1;
    //progress update here
    if (!(threadIdx.x || threadIdx.y)) //pretty sure this simplifies to both being 0
    {
        //atomicAdd((int*)progress, 1);
        //__threadfence_system();
        *progress += 1;
    }
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
    samplesPerPixel = 10;
    maxRecursion = 50;
    width = 266.6666666f;// 533.333333f;
    height = 150.0f;// 300.0f;
}

bool Raytracer::StartRender()
{
    std::cout << "Initializing Render...\n";
    
    int numPixels = width * height;

    //int deviceCount;
    //CheckCudaErrors(cudaGetDeviceCount(&deviceCount));
    //CheckCudaErrors(cudaDeviceSynchronize());

    //cudaDeviceProp properties;
    //CheckCudaErrors(cudaGetDeviceProperties(&properties, 0));

    CheckCudaErrors(cudaSetDevice(0));
    CheckCudaErrors(cudaDeviceSynchronize());

    curandState* randState;
    CheckCudaErrors(cudaMallocManaged(&randState, numPixels * sizeof(curandState)));

    size_t frameBufferSize = numPixels * sizeof(vec3);
    CheckCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
    int threadX = 8;
    int threadY = 8;

    /*clock_t start, stop;
    start = clock();*/
    

    dim3 blocks(width / threadX + 1, height / threadY + 1);
    dim3 threads(threadX, threadY);


    RenderInit << <blocks, threads >> > (width, height, randState);
    CheckCudaErrors(cudaDeviceSynchronize());

    //we have to allocate the shared memory manually because using unified memory here
        //causes an access violation for some reason
    volatile int* d_data, *h_data;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc((void**)&h_data, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer((int**)&d_data, (int*)h_data, 0);
    CheckCudaErrors(cudaDeviceSynchronize());


    *h_data = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    std::cout << "Initialized!\n";
    std::cout << "Rendering...\n";
    cudaEventRecord(start);
    Render <<<blocks, threads>>> (frameBuffer, width, height, samplesPerPixel, maxRecursion, mainCamera, entityList, numEntities, lights, numLights, d_data, randState);
    cudaEventRecord(stop);
    unsigned int numBlocks = blocks.x * blocks.y;
    float myProgress = 0.0f;
    int value = 0;
    std::cout << "Progress:\n";
    do 
    {
        cudaEventQuery(stop);//this query forces an updated read to our h_data value
        int value1 = *h_data;
        float renderProgress = (float)value1 / (float)numBlocks;
        if (renderProgress - myProgress > 0.1f)
        {
            std::cout << (int)(renderProgress * 100) << "% Complete...\n";
            myProgress = renderProgress;
        }


    } while (myProgress < 0.9f);

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    //stop = clock();
    //double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    float seconds;
    CheckCudaErrors(cudaEventElapsedTime(&seconds, start, stop));
    CheckCudaErrors(cudaDeviceSynchronize());
    seconds /= 1000; //milliseconds to seconds
    std::cerr << "Render took " << seconds << " seconds.\n";

    //FreeWorld << <1, 1 >> > (cam);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());


    //CheckCudaErrors(cudaFree(mesh));
    

    return false;
}

void Raytracer::WriteToFile()
{
    std::string extension = renderPath.substr(renderPath.find('.'));
    bool written = true;
    if (extension == ".ppm")
    {
        std::ofstream myFile(renderPath);
        // Output FB as PPM
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
    }
    else
    {
        int size = width * height * sizeof(vec3);
        unsigned char* data = new unsigned char[size];
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                size_t pixel_index = j * width + i;
                size_t read_index = j * width + (width - i);
                vec3 color = frameBuffer[(width * height) - read_index];//read the data backwards to flip it vertically
                unsigned char ir = unsigned char(255.99f * color.x);
                unsigned char ig = unsigned char(255.99f * color.y);
                unsigned char ib = unsigned char(255.99f * color.z);
                data[(pixel_index * 3) + 0] = ir;
                data[(pixel_index * 3) + 1] = ig;
                data[(pixel_index * 3) + 2] = ib;
            }
        }

        if (extension == ".png")
        {
            stbi_write_png(renderPath.c_str(), width, height, 3, data, 0);
        }
        else if (extension == ".jpg" || extension == ".jpeg")
        {
            stbi_write_jpg(renderPath.c_str(), width, height, 3, data, 100);
        }
        else if (extension == ".bmp")
        {
            stbi_write_bmp(renderPath.c_str(), width, height, 3, data);
        }
        else
        {
            std::cout << "Error writing file, output type not supported.";
            bool written = false;
        }

    }
    if(written)
        ShellExecute(0, 0, renderPath.c_str(), 0, 0, SW_SHOW);
    
    


    
    CheckCudaErrors(cudaFree(frameBuffer));
}
