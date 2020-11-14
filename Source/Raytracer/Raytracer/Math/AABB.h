#pragma once
#include "Ray.h"
#include "device_launch_parameters.h"




class AABB
{
public:
	__host__ __device__ AABB() {}
	__host__ __device__ AABB(const vec3& min, const vec3& max) : min(min), max(max) {}


	__host__ __device__ bool Hit(const Ray& r, float minDist, float maxDist)
	{
		for (int i = 0; i < 3; i++)
		{
			//first boundary
			float invertedDirection = 1.0f / r.direction[i];

			//float t01 = (min[i] - r.origin[i]) * invertedDirection;
			//float t02 = (max[i] - r.origin[i]) * invertedDirection;
			//float t0 = t01 < t02 ? t01 : t02;

			//float t11 = (min[i] - r.origin[i]) * invertedDirection;
			//float t12 = (max[i] - r.origin[i]) * invertedDirection;
			//float t1 = t01 > t02 ? t01 : t02;

			float t0 = (min[i] - r.origin[i]) * invertedDirection;
			float t1 = (max[i] - r.origin[i]) * invertedDirection;
			if (invertedDirection < 0.0f)
			{
				float temp = t0;
				t0 = t1;
				t1 = temp;
			}


			float tMin = (t0 > minDist) ? t0 : minDist;
			float tMax = (t1 < maxDist) ? t1 : maxDist;
			if (tMax <= tMin)
				return false;
		}
		return true;
	}


	

public:
	vec3 min;
	vec3 max;
};