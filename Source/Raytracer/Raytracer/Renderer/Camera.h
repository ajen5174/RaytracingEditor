#pragma once
#include "device_launch_parameters.h"
#include "../Math/Ray.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Core/Json.h"
class Camera
{
public:
	__device__ Camera::Camera()
	{
		
	}

	__host__ __device__ void SetView(vec3 lookFrom, vec3 lookAt, vec3 up)
	{
		float viewportHeight = 2.0f * tanf((fov * M_PI / 180.0f) / 2.0f);//measured in arbitrary units?
		float viewportWidth = aspectRatio * viewportHeight;
		
		vec3 w = Normalize(lookFrom - lookAt);
		vec3 u = Normalize(Cross(up, w));
		vec3 v = Cross(w, u);
		
		
		origin = lookFrom;
		horizontal = viewportWidth * u;
		vertical = viewportHeight * v;
		lowerLeft = origin - horizontal / 2 - vertical / 2 - w; //aims the camera down the z axis.
	}

	__device__ Ray Camera::GetRay(float u, float v)// u and v because they are normalized (0.0 - 1.0)
	{
		vec3 tempO = origin;
		vec3 templl = lowerLeft;
		vec3 tempv = vertical;
		vec3 temph = horizontal;
		return Ray(origin, (lowerLeft + u * horizontal + v * vertical) - origin);
	}

	bool Load(rapidjson::Value& value)
	{
		json::GetBool(value, "main", isMainCam);
		json::GetFloat(value, "fov", fov);
		json::GetFloat(value, "aspect", aspectRatio);


		return true;
	}

	

private:
	float aspectRatio;
	//float nearClip;
	//float farClip;
	float fov;
public:
	bool isMainCam;
private:
	vec3 origin;
	vec3 lowerLeft;
	vec3 vertical;
	vec3 horizontal;
};