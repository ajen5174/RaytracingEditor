#pragma once
#include "vec3.h"
#include "../Renderer/Hittable.h"

class Triangle : public Hittable
{
public:
	__host__ __device__ Triangle() : Hittable() {}
	__host__ __device__ Triangle(vec3 v1, vec3 v2, vec3 v3) 
		: Hittable()
	{
		vertices[0] = v1;
		vertices[1] = v2;
		vertices[2] = v3;
		normal = Cross(v2 - v1, v3 - v1).UnitVector();
		CreateBox();
	}
	__host__ __device__ Triangle(vec3 v1, vec3 v2, vec3 v3, vec3 normal)
		: Hittable()
	{
		vertices[0] = v1;
		vertices[1] = v2;
		vertices[2] = v3;
		this->normal = normal;
		CreateBox();
	}

private:
	__host__ __device__ void CreateBox()
	{
		vec3 averagePosition = (vertices[0] + vertices[1] + vertices[2]) / 3;

		vec3 min = vec3(1000000000.0f);

		for (int i = 0; i < 3; i++)
		{
			min.x = vertices[i].x < min.x ? vertices[i].x : min.x;
			min.y = vertices[i].y < min.y ? vertices[i].y : min.y;
			min.z = vertices[i].z < min.z ? vertices[i].z : min.z;
		}
		vec3 max = vec3(-1000000000.0f);

		for (int i = 0; i < 3; i++)
		{
			max.x = vertices[i].x > max.x ? vertices[i].x : max.x;
			max.y = vertices[i].y > max.y ? vertices[i].y : max.y;
			max.z = vertices[i].z > max.z ? vertices[i].z : max.z;
		}

		box = AABB(min, max);


		//float distances[3] = { (averagePosition - vertices[0]).Magnitude(), (averagePosition - vertices[1]).Magnitude(), (averagePosition - vertices[2]).Magnitude() };
		//int furthestIndex = 0;
		//float prevDist = 100000000000.0f;
		//for (int i = 0; i < 3; i++)
		//{
		//	if (distances[i] < prevDist)
		//	{
		//		prevDist = distances[i];
		//		furthestIndex = i;
		//	}
		//}

		//vec3 furth = vec3(fabs(vertices[furthestIndex].x), fabs(vertices[furthestIndex].y), fabs(vertices[furthestIndex].z));
		//box = AABB(averagePosition - furth, averagePosition + furth);
	}



public:
	// Inherited via Hittable
	virtual __host__ __device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) override
	{
		//using Moller-Trumbore algorithm
		vec3 O = ray.origin;
		vec3 D = ray.direction;

		vec3 A = vertices[0];
		vec3 B = vertices[1];
		vec3 C = vertices[2];

		vec3 T = O - A;
		vec3 E1 = B - A;
		vec3 E2 = C - A;

		vec3 P = Cross(D, E2);
		vec3 Q = Cross(T, E1);

		float determinant = Dot(P, E1);
		if (fabs(determinant) < FLT_EPSILON)
			return false;

		float scalar = 1.0f / determinant;

		float t = scalar * Dot(Q, E2);

		if (t > maxDist || t < minDist)
		{
			return false;
		}

		float u = scalar * Dot(P, T);
		float v = scalar * Dot(Q, D);
		float w = 1.0f - u - v;

		//if any barycentric coordinate is not normalized here then no collision occured
		if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f || w < 0.0f || w > 1.0f)
		{
			return false;
		}

		if (u + v + w > 1.0f)
		{
			return false;
		}

		vec3 intersectionPoint = (w * A) + (u * B) + (v * C);

		hitInfo.distance = t;
		hitInfo.u = u;
		hitInfo.v = v;
		hitInfo.w = w;
		hitInfo.point = intersectionPoint;
		hitInfo.material = material;
		hitInfo.normal = normal;
		return true;
	}

	__device__ virtual bool BoundingBox(AABB& outputBox) override
	{
		outputBox.min.x = box.min.x;
		outputBox.min.y = box.min.y;
		outputBox.min.z = box.min.z;

		outputBox.max.x = box.max.x;
		outputBox.max.y = box.max.y;
		outputBox.max.z = box.max.z;
		return true;
	}


public:
	vec3 vertices[3];

	//not sure if this is strictly needed, or if it can be replaced with a method call.
	//vec3 averagePosition;
	Material* material = nullptr;
	vec3 normal;
	AABB box;
};