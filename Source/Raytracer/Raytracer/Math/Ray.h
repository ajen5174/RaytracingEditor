#pragma once
#include "device_launch_parameters.h"
#include "vec3.h"

__device__ inline float schlick(float cosine, float index)
{
	return 0.0f;
}

__device__ inline bool refract(const vec3& direction, const vec3& normal, float niOverNt, vec3& refracted)
{
	return false;
}

__device__ inline vec3 reflect(const vec3& direction, const vec3& normal)
{
	return direction - 2.0f * Dot(direction, normal) * normal;
}




class Ray
{
public:
	__device__ Ray() : origin(vec3(0.0f)), direction(vec3(0.0f)) {}
	__device__ Ray(vec3& origin, vec3& direction) : origin(origin), direction(direction) {}

	__device__ vec3 PointAt(float t)
	{
		return origin + (direction * t);
	}


public:
	vec3 origin;
	vec3 direction;
};

