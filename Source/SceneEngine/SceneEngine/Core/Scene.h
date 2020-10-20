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
	Entity* Remove(Entity* entity, bool destroy = true);
	Entity* GetEntityByName(uint32_t name);
	Entity* GetEntityByName(StringId& name);
	Entity* GetEntityByFloatId(float id);

	Camera* GetMainCamera() { return mainCamera; }

public:
	Camera* mainCamera;

private:
	std::vector<Entity*> entities;

};
#endif