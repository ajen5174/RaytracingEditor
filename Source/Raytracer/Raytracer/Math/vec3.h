#pragma once
#include <math.h>
#include <iostream>
#include "device_launch_parameters.h"

struct vec3
{
	__host__ __device__ vec3() : x(0), y(0), z(0) {}
	__host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__ vec3(float n) : x(n), y(n), z(n) {}

	__host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__ float operator[](int i) const { return (&x)[i]; }
	__host__ __device__ float& operator[](int i) { return (&x)[i]; }

	__host__ __device__ float Magnitude() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ float SqrMagnitude() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ void Normalize();

	float x, y, z;
}; 

//std::istream& operator>>(std::istream& is, vec3& v) {
//	is >> v.x >> v.y >> v.z;
//	return is;
//}
//
//std::ostream& operator<<(std::ostream& os, const vec3& v) {
//	os << v.x << " " << v.y << " " << v.z;
//	return os;
//}



__host__ __device__ vec3 operator+(const vec3& v1, const vec3& v2);


__host__ __device__ vec3 operator-(const vec3& v1, const vec3& v2);


//vec3 operator*(const vec3& v1, const vec3& v2)
//{
//	return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
//}

__host__ __device__ vec3 operator/(const vec3& v1, const vec3& v2);


__host__ __device__ vec3 operator*(float t, const vec3& v2);


__host__ __device__ vec3 operator*(const vec3& v2, float t);


__host__ __device__ vec3 operator/(const vec3& v2, float t);


__host__ __device__ float Dot(const vec3& v1, const vec3 v2);


__host__ __device__ vec3 Cross(const vec3& v1, const vec3& v2);


__host__ __device__ vec3 Normalize(vec3 v);
