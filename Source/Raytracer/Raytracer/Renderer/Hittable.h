#pragma once
#include "../Math/Ray.h"
#include "Material.h"

struct HitInfo
{
	float distance;
	vec3 point;
	vec3 normal;
	Material* material;
};

class Hittable
{
public:
	__device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) = 0;
};