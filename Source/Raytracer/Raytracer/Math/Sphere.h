#pragma once
#include "../Renderer/Hittable.h"

class Sphere : public Hittable
{
public:
	__device__ Sphere(vec3& center, float radius) : center(center), radius(radius) {}


public:
	vec3 center;
	float radius;

	// Inherited via Hittable
	__host__ __device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) override
	{
		vec3 sphereToOrigin = ray.origin - center;
		float a = Dot(ray.direction, ray.direction);
		float b = 2.0f * Dot(ray.direction, sphereToOrigin);
		float c = Dot(sphereToOrigin, sphereToOrigin) - radius * radius;

		float discriminant = b * b - 4.0f * a * c;

		
		if (discriminant > 0.0f)
		{
			//two hits
			float temp = (-b - sqrtf(discriminant)) / 2.0f * a;
			if (temp < maxDist && temp > minDist)
			{
				hitInfo.distance = temp;
				hitInfo.point = ray.PointAt(temp);
				hitInfo.normal = (hitInfo.point - center) / radius;
				hitInfo.material = material;
				return true;
			}
			temp = (-b + sqrtf(discriminant)) / 2.0f * a;
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
	//Texture* texture;
	Material* material;
};
