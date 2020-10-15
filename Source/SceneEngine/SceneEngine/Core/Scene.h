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
	void Add(Entity* entity);
	Entity* Remove(Entity* entity, bool destroy = true);
	Entity* GetEntityByName(StringId& name);

	Camera* GetMainCamera() { return mainCamera; }

private:
	std::vector<Entity*> entities;
	Camera* mainCamera;

};
#endif