#pragma once
#include "device_launch_parameters.h"
#include "vec3.h"

__device__ float schlick(float cosine, float index);

__device__ bool refract(const vec3& direction, const vec3& normal, float niOverNt, vec3& refracted);

__device__ vec3 reflect(const vec3& direction, const vec3& normal);


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

