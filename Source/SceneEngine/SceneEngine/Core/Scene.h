#include <vector>
#include "StringId.h"
#include "Entity.h"
#include "../Renderer/Camera.h"

#ifndef SCENE_H
#define SCENE_H
class Scene
{
public:
	void Update();
	void Draw();
	void DrawPick();
	void Add(Entity* entity);
	void Deselect();
	void Load(rapidjson::Value& value);
	void BuildJSON(rapidjson::Document& doc);

	Entity* Remove(Entity* entity, bool destroy = true);
	Entity* GetEntityByName(uint32_t name);
	Entity* GetEntityByName(StringId& name);
	Entity* GetEntityByFloatId(float id);
	std::vector<Entity*> GetEntities() { return entities; }
	template <typename T>
	std::vector<T*>* Get();

	Camera* GetMainCamera() { return mainCamera; }

public:
	Camera* mainCamera;

private:
	std::vector<Entity*> entities;

};
#endif

template<typename T>
inline std::vector<T*>* Scene::Get()
{
	std::vector<T*>* components = new std::vector<T*>();
	for (Entity* e : entities)
	{
		T* temp = e->GetComponent<T>();
		if (temp)
			components->push_back(temp);
	}
	return components;
}
