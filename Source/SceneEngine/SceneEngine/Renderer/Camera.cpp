#include "Camera.h"
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

#include "../EngineLibrary.h"


Camera::Camera(StringId& name, Entity* owner)
	:Component(name, owner)
{

}

void Camera::SetProjection(float fov, float aspectRatio, float nearClip, float farClip)
{
	this->fov = fov;
	this->aspectRatio = aspectRatio;
	this->nearClip = nearClip;
	this->farClip = farClip;

	projectionMatrix = glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip);
}

void Camera::Update()
{
	Transform* transform = owner->GetComponent<Transform>();
	glm::vec3 target = transform->translation + (transform->rotation * glm::vec3(0.0f, 0.0f, 1.0f));

	viewMatrix = glm::lookAt(transform->translation, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

bool Camera::Load(const rapidjson::Value&)
{
	return false;
}

void Camera::Destroy()
{
}

void Camera::Initialize()
{
}
