#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "Renderer/Hittable.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include "Math/vec3.h"
#include "Renderer/Camera.h"


#define CheckCudaErrors(val) CheckCuda( (val), #val, __FILE__, __LINE__ )
void CheckCuda(cudaError_t result, char const* const func, const char* const file, int const line);

class Raytracer
{
public:
	Raytracer(std::string sceneToLoad, std::string renderPath);

	bool StartRender();

	void WriteToFile();

private:
	bool LoadScene(std::string sceneToLoad);

public:
	int width;
	int height;
	vec3* frameBuffer;

private:
	std::string renderPath;
	std::vector<Hittable*> hittables;
	//std::vector<Light*> lights;
};
