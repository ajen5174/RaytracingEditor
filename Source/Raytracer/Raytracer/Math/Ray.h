#pragma once
#include "device_launch_parameters.h"
#include "vec3.h"
#include <curand_kernel.h>



__device__ inline float Schlick(float cosine, float index)
{
	float r0 = (1 - index) / (1 + index);
	r0 = r0 * r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

__device__ inline bool Refract(const vec3& direction, const vec3& normal, float niOverNt, vec3& refracted)
{
	//using snells 
	vec3 normalizedDir = Normalize(direction);
	float dt = Dot(normalizedDir, normal);
	float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
	if (discriminant > 0.0f)
	{
		refracted = niOverNt * (normalizedDir - normal * dt) - normal * sqrtf(discriminant);
		return true;
	}
	else
	{
		return false;
	}

	/*auto cosTheta = fmin(Dot(-direction, normal), 1.0f);
	vec3 rOutPerp = niOverNt * (direction + cosTheta * normal);
	vec3 rOutParallel = -sqrtf(fabs(1.0f - rOutPerp.SqrMagnitude())) * normal;
	refracted =  rOutPerp + rOutParallel;*/
}

__device__ inline vec3 Reflect(const vec3& direction, const vec3& normal)
{
	return direction - 2.0f * Dot(direction, normal) * normal;
}

__device__ inline vec3 RandomInUnitSphere(curandState* localRandState)
{
	vec3 p;
	do 
	{
		p = (2.0f * (RANDVEC3)) - vec3(1.0f);
	} while (p.SqrMagnitude() >= 1.0f);
	return p;
}

__device__ inline vec3 RandomInHemisphere(curandState* localRandState, const vec3& normal)
{
	vec3 p = RandomInUnitSphere(localRandState);
	if (Dot(p, normal) > 0.0f)
		return p;
	else
		return -p;
}



class Ray
{
public:
	__host__ __device__ Ray() : origin(vec3(0.0f)), direction(vec3(0.0f)) {}
	__host__ __device__ Ray(vec3& origin, vec3& direction) : origin(origin), direction(direction) {}

	__host__ __device__ vec3 PointAt(float t) const
	{
		return origin + (direction * t);
	}


public:
	vec3 origin;
	vec3 direction;
};

