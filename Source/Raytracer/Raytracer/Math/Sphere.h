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
	__device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) override
	{
		return false;
	}
	//Texture* texture;
	//Material* material;
};
