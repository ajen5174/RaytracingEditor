#include "Ray.h"

__device__ float schlick(float cosine, float index)
{
	return 0.0f;
}

__device__ bool refract(const vec3& direction, const vec3& normal, float niOverNt, vec3& refracted)
{
	return false;
}

__device__ vec3 reflect(const vec3& direction, const vec3& normal)
{
	return direction - 2.0f * Dot(direction, normal) * normal;
}
