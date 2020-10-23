#pragma once
#include "../Math/Ray.h"
#include "Material.h"

struct HitInfo
{
	float distance;
	vec3 point;
	vec3 normal;
	Material* material;
	float u = -1.0f, v = -1.0f, w = -1.0f; //for triangle barycentric coordinates
};

class Hittable
{
public:
	__host__ __device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) = 0;
};