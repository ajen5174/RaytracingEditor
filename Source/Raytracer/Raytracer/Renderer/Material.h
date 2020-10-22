#pragma once
#include "Hittable.h"

class Material
{
public:
	//virtual bool Scatter(Ray& ray, HitInfo& hitinfo, vec3 color, Ray& bouncedRay) = 0;
};

class Lambertian : Material
{
public:
	/*virtual bool Scatter(Ray& ray, HitInfo& hitinfo, vec3 color, Ray& bouncedRay) override
	{

	}*/

public:
	vec3 albedo;
};