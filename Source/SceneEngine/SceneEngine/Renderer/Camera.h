#pragma once
#include "../Components/Component.h"
#include <glm/glm/glm.hpp>
#include "../Core/Entity.h"

class Camera : public Component
{
public:
	Camera(StringId& name, Entity* owner);
	void SetProjection(float fov, float aspectRatio, float nearClip, float farClip);
	// Inherited via Component
	virtual void Update() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void Destroy() override;
	virtual void Initialize() override;
	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) override;

public:
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;

private:
	float fov;
	float aspectRatio;
	float nearClip;
	float farClip;








};
