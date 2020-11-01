#pragma once
#include "vec3.h"



class mat4 
{
public:
	__host__ __device__ mat4() 
	{
		data[0][0] = 1.0f;
		data[1][0] = 0.0f;
		data[2][0] = 0.0f;
		data[3][0] = 0.0f;

		data[0][1] = 0.0f;
		data[1][1] = 1.0f;
		data[2][1] = 0.0f;
		data[3][1] = 0.0f;

		data[0][2] = 0.0f;
		data[1][2] = 0.0f;
		data[2][2] = 1.0f;
		data[3][2] = 0.0f;
		
		data[0][3] = 0.0f;
		data[1][3] = 0.0f;
		data[2][3] = 0.0f;
		data[3][3] = 1.0f;
	}
	__host__ __device__ inline mat4(const vec3& translation, const vec3& rotation, const vec3& scale);


	

public:
	float data[4][4];
};

__host__ __device__ inline vec3 operator*(const mat4& m1, const vec3& v1)
{
	vec3 result(0.0f);

	result.x = m1.data[0][0] * v1.x +
			   m1.data[1][0] * v1.y +
			   m1.data[2][0] * v1.z +
			   m1.data[3][0] * 1.0f;

	result.y = m1.data[0][1] * v1.x +
			   m1.data[1][1] * v1.y +
			   m1.data[2][1] * v1.z +
			   m1.data[3][1] * 1.0f;

	result.z = m1.data[0][2] * v1.x +
			   m1.data[1][2] * v1.y +
			   m1.data[2][2] * v1.z +
			   m1.data[3][2] * 1.0f;

	//omit multiplication for the last column, as it is not needed for this application, and it will result in 1 anyway

	return result;
}
__host__ __device__ inline mat4 operator*(const mat4& m1, const mat4& m2)
{
	mat4 m;
	//row 1
	m.data[0][0] = m1.data[0][0] * m2.data[0][0] +
				   m1.data[1][0] * m2.data[0][1] +
				   m1.data[2][0] * m2.data[0][2] +
				   m1.data[3][0] * m2.data[0][3];

	m.data[1][0] = m1.data[0][0] * m2.data[1][0] +
				   m1.data[1][0] * m2.data[1][1] +
				   m1.data[2][0] * m2.data[1][2] +
				   m1.data[3][0] * m2.data[1][3];

	m.data[2][0] = m1.data[0][0] * m2.data[2][0] +
				   m1.data[1][0] * m2.data[2][1] +
				   m1.data[2][0] * m2.data[2][2] +
				   m1.data[3][0] * m2.data[2][3];

	m.data[3][0] = m1.data[0][0] * m2.data[3][0] +
				   m1.data[1][0] * m2.data[3][1] +
				   m1.data[2][0] * m2.data[3][2] +
				   m1.data[3][0] * m2.data[3][3];

	//row 2
	m.data[0][1] = m1.data[0][1] * m2.data[0][0] +
				   m1.data[1][1] * m2.data[0][1] +
				   m1.data[2][1] * m2.data[0][2] +
				   m1.data[3][1] * m2.data[0][3];

	m.data[1][1] = m1.data[0][1] * m2.data[1][0] +
				   m1.data[1][1] * m2.data[1][1] +
				   m1.data[2][1] * m2.data[1][2] +
				   m1.data[3][1] * m2.data[1][3];

	m.data[2][1] = m1.data[0][1] * m2.data[2][0] +
				   m1.data[1][1] * m2.data[2][1] +
				   m1.data[2][1] * m2.data[2][2] +
				   m1.data[3][1] * m2.data[2][3];

	m.data[3][1] = m1.data[0][1] * m2.data[3][0] +
				   m1.data[1][1] * m2.data[3][1] +
				   m1.data[2][1] * m2.data[3][2] +
				   m1.data[3][1] * m2.data[3][3];

	//row 3
	m.data[0][2] = m1.data[0][2] * m2.data[0][0] +
				   m1.data[1][2] * m2.data[0][1] +
				   m1.data[2][2] * m2.data[0][2] +
				   m1.data[3][2] * m2.data[0][3];
							  
	m.data[1][2] = m1.data[0][2] * m2.data[1][0] +
				   m1.data[1][2] * m2.data[1][1] +
				   m1.data[2][2] * m2.data[1][2] +
				   m1.data[3][2] * m2.data[1][3];
							  
	m.data[2][2] = m1.data[0][2] * m2.data[2][0] +
				   m1.data[1][2] * m2.data[2][1] +
				   m1.data[2][2] * m2.data[2][2] +
				   m1.data[3][2] * m2.data[2][3];
							  
	m.data[3][2] = m1.data[0][2] * m2.data[3][0] +
				   m1.data[1][2] * m2.data[3][1] +
				   m1.data[2][2] * m2.data[3][2] +
				   m1.data[3][2] * m2.data[3][3];

	//row 4
	m.data[0][3] = m1.data[0][3] * m2.data[0][0] +
				   m1.data[1][3] * m2.data[0][1] +
				   m1.data[2][3] * m2.data[0][2] +
				   m1.data[3][3] * m2.data[0][3];

	m.data[1][3] = m1.data[0][3] * m2.data[1][0] +
				   m1.data[1][3] * m2.data[1][1] +
				   m1.data[2][3] * m2.data[1][2] +
				   m1.data[3][3] * m2.data[1][3];
							  
	m.data[2][3] = m1.data[0][3] * m2.data[2][0] +
				   m1.data[1][3] * m2.data[2][1] +
				   m1.data[2][3] * m2.data[2][2] +
				   m1.data[3][3] * m2.data[2][3];
							  
	m.data[3][3] = m1.data[0][3] * m2.data[3][0] +
				   m1.data[1][3] * m2.data[3][1] +
				   m1.data[2][3] * m2.data[3][2] +
				   m1.data[3][3] * m2.data[3][3];

	return m;
}

//I believe I need to do column major for this to make sense, so I think this is right
__host__ __device__ inline mat4 TranslationMatrix(const vec3& translation)
{
	mat4 m;
	m.data[3][0] = translation.x;
	m.data[3][1] = translation.y;
	m.data[3][2] = translation.z;
	m.data[3][3] = 1.0f;
	return m;
}

__host__ __device__ inline mat4 ScaleMatrix(const vec3& scale)
{
	mat4 m;
	m.data[0][0] = scale.x;
	m.data[1][1] = scale.y;
	m.data[2][2] = scale.z;
	m.data[3][3] = 1.0f;
	return m;
}

__host__ __device__ inline mat4 RotationMatrix(const vec3& rotation)
{
	float converter = M_PI / 180.0f;
	mat4 x;
	x.data[0][0] = 1.0f;
	x.data[1][1] = cosf(rotation.x * converter);
	x.data[1][2] = sinf(rotation.x * converter);
	x.data[2][1] = -sinf(rotation.x * converter);
	x.data[2][2] = cosf(rotation.x * converter);
	x.data[3][3] = 1.0f;


	mat4 y;
	y.data[0][0] = cosf(rotation.y * converter);
	y.data[1][1] = 1.0f;
	y.data[0][2] = -sinf(rotation.y * converter);
	y.data[2][0] = sinf(rotation.y * converter);
	y.data[2][2] = cosf(rotation.y * converter);
	y.data[3][3] = 1.0f;

	mat4 z;
	z.data[0][0] = cosf(rotation.z * converter);
	z.data[0][1] = sinf(rotation.z * converter);
	z.data[1][0] = -sinf(rotation.z * converter);
	z.data[1][1] = cosf(rotation.z * converter);
	z.data[2][2] = 1.0f;
	z.data[3][3] = 1.0f;

	return z * y * x;
}







__host__ __device__ inline mat4::mat4(const vec3& translation, const vec3& rotation, const vec3& scale)
{
	{
		mat4 t = TranslationMatrix(translation);
		mat4 r = RotationMatrix(rotation);
		mat4 s = ScaleMatrix(scale);

		mat4 m = (t * r) * s;
		data[0][0] = m.data[0][0];
		data[1][0] = m.data[1][0];
		data[2][0] = m.data[2][0];
		data[3][0] = m.data[3][0];

		data[0][1] = m.data[0][1];
		data[1][1] = m.data[1][1];
		data[2][1] = m.data[2][1];
		data[3][1] = m.data[3][1];

		data[0][2] = m.data[0][2];
		data[1][2] = m.data[1][2];
		data[2][2] = m.data[2][2];
		data[3][2] = m.data[3][2];

		data[0][3] = m.data[0][3];
		data[1][3] = m.data[1][3];
		data[2][3] = m.data[2][3];
		data[3][3] = m.data[3][3];

	}
}