#pragma once
#include "Hittable.h"

class Material
{
public:
	__device__ bool Scatter(Ray& ray, HitInfo& hitInfo, vec3& color, Ray& bouncedRay, curandState* localRandState)
	{
		if (materialType == 'l')
		{
			//lambert stuff
			vec3 target = hitInfo.point + RandomInHemisphere(localRandState, hitInfo.normal);
			bouncedRay = Ray(hitInfo.point, target - hitInfo.point);

			color = albedo;
			return true;
		}
		else if (materialType == 'm')
		{
			vec3 reflected = Reflect(Normalize(ray.direction), hitInfo.normal);
			bouncedRay = Ray(hitInfo.point, reflected + fuzz * Normalize(RandomInUnitSphere(localRandState)));
			color = albedo;
			return (Dot(bouncedRay.direction, hitInfo.normal) > 0.0f);
		}
		else if (materialType == 'd')
		{
			color = vec3(1.0f);
			vec3 outwardNormal;
			//vec3 reflected = Reflect(ray.direction, hitInfo.normal);
			float niOverNt;
			if (Dot(ray.direction, hitInfo.normal) > 0.0f) // if we hit the back of the object
			{
				outwardNormal = -hitInfo.normal;
				niOverNt = refractionIndex;
				//cosine = Dot(ray.direction, hitInfo.normal) / ray.direction.Magnitude();
				//cosine = sqrtf(1.0f - refractionIndex * refractionIndex * (1 - cosine * cosine));
			}
			else
			{
				outwardNormal = hitInfo.normal;
				niOverNt = 1.0f / refractionIndex;
				//cosine = -Dot(ray.direction, hitInfo.normal) / ray.direction.Magnitude();
			}
			vec3 unitDirection = ray.direction.UnitVector();
			float cosTheta = fmin(Dot(-unitDirection, outwardNormal), 1.0f);
			float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

			bool cannotRefract = niOverNt * sinTheta > 1.0f;
			vec3 direction;
			if (cannotRefract)
				direction = Reflect(unitDirection, outwardNormal);
			else
				direction = Refract(unitDirection, outwardNormal, niOverNt);
			
			bouncedRay = Ray(hitInfo.point, direction);
			return true;
		}
		
	}

public:
	vec3 albedo;
	float fuzz;
	float refractionIndex;
	char materialType;
};

//class Lambertian : public Material
//{
//public:
//	__host__ __device__ Lambertian() : albedo(0.0f) {}
//
//	// Inherited via Material
//	virtual __host__ __device__ bool Scatter(Ray& ray, HitInfo& hitInfo, vec3& color, Ray& bouncedRay) override
//	
//
//public:
//	vec3 albedo;
//
//	
//};