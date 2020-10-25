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
	}
	__host__ __device__ Triangle(vec3 v1, vec3 v2, vec3 v3, vec3 normal)
	{
		vertices[0] = v1;
		vertices[1] = v2;
		vertices[2] = v3;
		this->normal = normal;
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
		//hitInfo.material = material;
		hitInfo.normal = normal;
		return true;
	}

public:
	vec3 vertices[3];

	//not sure if this is strictly needed, or if it can be replaced with a method call.
	//vec3 averagePosition;
	//Material* material;
	vec3 normal;
};