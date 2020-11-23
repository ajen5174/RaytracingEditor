#pragma once
#include "Components/Component.h"
#include "glm/glm.hpp"
#include "Renderer/Shader.h"
#include "Core/Entity.h"
#include "Components/Transform.h"



class Light : public Component
{
public:

	Light(StringId& name, Entity* owner) : Component(name, owner) {}

	// Inherited via Component
	virtual void Destroy() override;
	virtual bool Load(const rapidjson::Value&) override;
	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) override;
	virtual void Initialize() override;
	virtual void Update() override;

	void SetShader(std::string lightname, Shader* shader);


public:
	float intensity;
	glm::vec3 color;
	glm::vec3 direction;
	std::string lightType = "point";
};