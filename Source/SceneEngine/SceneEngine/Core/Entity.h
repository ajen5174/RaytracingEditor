#include "../Components/Transform.h"
#include <vector>
#include "Object.h"
#include "../Components/RenderComponent.h"

class Scene;

#ifndef ENTITY_H
#define ENTITY_H
class Entity : public Object
{
	
public:
	Entity(StringId& name) 
		: parent(nullptr), scene(nullptr), selected(false), Object(name) 
	{
		if (!GetComponent<Transform>())
		{
			Transform* transform = new Transform(name, this);
			AddComponent(transform);
		}
	}

	template <typename T>
	T* GetComponent()
	{
		for (int i = 0; i < components.size(); i++)
		{
			T* component = dynamic_cast<T*>(components[i]);
			if (component != nullptr)
			{
				return component;
			}
		}

		return nullptr;
	}

	void Update();
	void Draw();
	void DrawPick();
	void AddComponent(Component* component);
	void RemoveComponent(Component* component);
	std::vector<Entity*> GetChildren();

	Scene* GetScene() { return scene; };

	int GetFloatData(uint32_t flags, float* data, int size);

	// Inherited via Object
	virtual void Destroy() override;

	virtual bool Load(const rapidjson::Value&) override;
	virtual void BuildJSON(rapidjson::Value&, rapidjson::MemoryPoolAllocator<>&) override;

	virtual void Initialize() override;

private:
	bool LoadComponents(const rapidjson::Value& value);

public:
	//Transform transform;
	//Transform localTransform;
	bool selected = false;
	Scene* scene;

private:
	Entity* parent;
	std::vector<Component*> components;
	std::vector<Entity*> children;


	

};
#endif