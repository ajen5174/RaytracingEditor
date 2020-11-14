#pragma once
#include "../Math/Ray.h"
#include "../Math/AABB.h"
class Material;

struct HitInfo
{
	float distance = -1.0f;
	vec3 point;
	vec3 normal;
	Material* material = nullptr;
	float u = -1.0f, v = -1.0f, w = -1.0f; //for triangle barycentric coordinates
};

__device__ inline AABB SurroundingBox(AABB box1, AABB box2)
{
	vec3 small(fmin(box1.min.x, box2.min.x),
			   fmin(box1.min.y, box2.min.y), 
			   fmin(box1.min.z, box2.min.z));

	vec3 big(fmax(box1.max.x, box2.max.x),
			 fmax(box1.max.y, box2.max.y),
			 fmax(box1.max.z, box2.max.z));

	return AABB(small, big);
}

class Hittable
{
public:
	__host__ __device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) = 0;
	__device__ virtual bool BoundingBox(AABB& outputBox) = 0;
};