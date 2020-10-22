#pragma once
#include "../Renderer/Hittable.h"

class Sphere : Hittable
{
public:
	Sphere(vec3& center, float radius) : center(center), radius(radius) {}


public:
	vec3 center;
	float radius;

	// Inherited via Hittable
	virtual bool Hit(Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) override;
	//Texture* texture;
	//Material* material;
};
