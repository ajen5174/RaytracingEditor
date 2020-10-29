#include "Camera.h"
#include "../Core/Scene.h"
#include <glm/glm/glm.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

#include "../EngineLibrary.h"
#include "../Core/Json.h"


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

bool Camera::Load(const rapidjson::Value& value)
{
	bool mainCamera;
	json::GetBool(value, "main", mainCamera);
	if (mainCamera)
	{
		owner->scene->mainCamera = this;
	}

	json::GetFloat(value, "fov", fov);

	json::GetFloat(value, "aspect", aspectRatio);

	SetProjection(fov, aspectRatio, 0.01f, 100.0f);

	return true;
}

void Camera::Destroy()
{
}

void Camera::Initialize()
{
}

void Camera::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildCString(v, "type", "Camera", mem);
	if (owner->scene->mainCamera == this)
	{
		json::BuildBool(v, "main", true, mem);
	}

	json::BuildFloat(v, "fov", fov, mem);
	json::BuildFloat(v, "aspect", aspectRatio, mem);
}
