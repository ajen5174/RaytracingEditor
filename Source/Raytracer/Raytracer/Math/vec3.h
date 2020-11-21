#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include "device_launch_parameters.h"

struct vec3
{
	__host__ __device__ vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	__host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__ vec3(float n) : x(n), y(n), z(n) {}

	__host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__ float operator[](int i) const { return (&x)[i]; }
	__host__ __device__ float& operator[](int i) { return (&x)[i]; }

	__host__ __device__ float Magnitude() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ float SqrMagnitude() const { return (x * x + y * y + z * z); }
	__host__ __device__ vec3 UnitVector()
	{
		float k = 1.0f / sqrtf(x * x + y * y + z * z);
		//x *= k;
		//y *= k;
		//z *= k;
		return vec3(x * k, y * k, z * k);
	}

	float x, y, z;
}; 

inline std::istream& operator>>(std::istream& is, vec3& v)
{
	is >> v.x >> v.y >> v.z;
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& v)
{
	os << v.x << " " << v.y << " " << v.z;
	return os;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v2)
{
	return vec3(t * v2.x, t * v2.y, t * v2.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v2, float t)
{
	return vec3(t * v2.x, t * v2.y, t * v2.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

__host__ __device__ inline vec3 operator/(const vec3& v2, float t)
{
	float k = 1 / t;
	return k * v2;
}

__host__ __device__ inline float Dot(const vec3& v1, const vec3 v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

__host__ __device__ inline vec3 Cross(const vec3& v1, const vec3& v2)
{
	return vec3((v1.y * v2.z - v1.z * v2.y),
		(-(v1.x * v2.z - v1.z * v2.x)),
		(v1.x * v2.y - v1.y * v2.x));
}

__host__ __device__ inline vec3 Normalize(vec3 v)
{
	return v / v.Magnitude();
}

#define RANDVEC3 vec3(curand_uniform(localRandState),curand_uniform(localRandState),curand_uniform(localRandState))
