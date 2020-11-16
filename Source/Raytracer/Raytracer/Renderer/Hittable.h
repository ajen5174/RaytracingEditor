#pragma once
#include "../Math/Ray.h"
class Material;
class AABB;

struct HitInfo
{
	float distance = -1.0f;
	vec3 point;
	vec3 normal;
	Material* material = nullptr;
	float u = -1.0f, v = -1.0f, w = -1.0f; //for triangle barycentric coordinates
};


class Hittable
{
public:
	__host__ __device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) = 0;
	__host__ __device__ virtual bool BoundingBox(AABB& outputBox) = 0;
};