#include "vec3.h"


__host__ __device__ void vec3::Normalize()
{
	float k = 1.0f / sqrtf(x * x + y * y + z * z);
	x *= k;
	y *= k;
	z *= k;
}

__host__ __device__ vec3 operator+(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ vec3 operator-(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__host__ __device__ vec3 operator/(const vec3& v1, const vec3& v2)
{
	return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

__host__ __device__ vec3 operator*(float t, const vec3& v2)
{
	return vec3(t * v2.x, t * v2.y, t * v2.z);
}

__host__ __device__ vec3 operator*(const vec3& v2, float t)
{
	return vec3(t * v2.x, t * v2.y, t * v2.z);
}

__host__ __device__ vec3 operator/(const vec3& v2, float t)
{
	float k = 1 / t;
	return k * v2;
}

__host__ __device__ float Dot(const vec3& v1, const vec3 v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

__host__ __device__ vec3 Cross(const vec3& v1, const vec3& v2)
{
	return vec3((v1.y * v2.z - v1.z * v2.y),
		(-(v1.x * v2.z - v1.z * v2.x)),
		(v1.x * v2.y - v1.y * v2.x));
}

__host__ __device__ vec3 Normalize(vec3 v)
{
	return v / v.Magnitude();
}