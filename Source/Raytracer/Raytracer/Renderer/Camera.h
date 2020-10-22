#pragma once
#include "device_launch_parameters.h"
#include "../Math/Ray.h"
#define _USE_MATH_DEFINES
#include <math.h>
class Camera
{
public:
	__device__ Camera::Camera(vec3 origin, vec3 lookat, vec3 up, float fov, float aspectRatio)
	{
		float viewportHeight = 2.0f * tanf(fov * M_PI / 180.0f);//measured in arbitrary units?
		float viewportWidth = aspectRatio * viewportHeight;
		this->aspectRatio = aspectRatio;
		origin = origin;
		horizontal = vec3(viewportWidth, 0.0f, 0.0f);
		vertical = vec3(0.0f, viewportHeight, 0.0f);
		lowerLeft = vec3(origin - horizontal / 2 - vertical / 2 - vec3(0.0f, 0.0f, 1.0f)); //aims the camera down the z axis.
	}

	__device__ Ray Camera::GetRay(float u, float v)// u and v because they are normalized (0.0 - 1.0)
	{
		return Ray(origin, (lowerLeft + u * horizontal + v * vertical) - origin);
	}

	

private:
	float aspectRatio;
	float nearClip;
	float farClip;

	vec3 origin;
	vec3 lowerLeft;
	vec3 vertical;
	vec3 horizontal;
};