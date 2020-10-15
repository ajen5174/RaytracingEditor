#include "Camera.h"

void Camera::SetProjection(float fov, float aspectRatio, float nearClip, float farClip)
{
	this->fov = fov;
	this->aspectRatio = aspectRatio;
	this->nearClip = nearClip;
	this->farClip = farClip;
}

void Camera::Update()
{
}
