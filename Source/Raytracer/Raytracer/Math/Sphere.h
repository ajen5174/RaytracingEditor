#pragma once
#include "AABB.h"

class Sphere
{
public:

	__host__ __device__ Sphere() {}
	__host__ __device__ Sphere(vec3& center, float radius)
	{
		this->center = center;
		this->radius = radius;
	}


	__host__ __device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const 
	{
		vec3 sphereToOrigin = ray.origin - center;
		float a = Dot(ray.direction, ray.direction);
		float b = Dot(ray.direction, sphereToOrigin);
		float c = Dot(sphereToOrigin, sphereToOrigin) - radius * radius;

		float discriminant = b * b - a * c;

		
		if (discriminant > 0.0f)
		{
			//two hits
			float temp = (-b - sqrtf(discriminant)) / a;
			if (temp < maxDist && temp > minDist)
			{
				hitInfo.distance = temp;
				hitInfo.point = ray.PointAt(temp);
				hitInfo.normal = (hitInfo.point - center) / radius;
				hitInfo.material = material;
				return true;
			}
			temp = (-b + sqrtf(discriminant)) / a;
			if (temp < maxDist && temp > minDist)
			{
				hitInfo.distance = temp;
				hitInfo.point = ray.PointAt(temp);
				hitInfo.normal = (hitInfo.point - center) / radius;
				hitInfo.material = material;
				return true;
			}
		}

		return false;

	}

	__host__ __device__  bool BoundingBox(AABB& outputBox) 
	{
		outputBox = AABB(center - vec3(radius), center + vec3(radius));
		return true;
	}

public:
	vec3 center;
	float radius = 1.0f;
	//Texture* texture;
	Material* material = nullptr;
};
