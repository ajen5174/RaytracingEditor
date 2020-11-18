#include "../Core/Object.h"
#ifndef COMPONENT_H
#define COMPONENT_H
class Entity;

//flags
enum class ComponentType
{
	NONE = 0,
	TRANSFORM,
	LIGHT,
	MODEL_RENDER,
	CAMERA,
	MATERIAL,
	SPHERE
};

class Component : public Object
{
public:
	

public:
	Component(StringId& name, Entity* owner) : owner(owner), Object(name) {}

	virtual void Update() = 0;
	Entity* GetOwner() { return owner; }
	void SetOwner(Entity* newOwner) { owner = newOwner; };

	ComponentType GetComponentType() { return componentType; }

	// Inherited via Object
	//virtual void Destroy() override;

	//virtual bool Load(const rapidjson::Value&) override;

	//virtual void Initialize() override;

private:
	ComponentType componentType = ComponentType::NONE;

protected:
	Entity* owner = nullptr;


	

};
#endif