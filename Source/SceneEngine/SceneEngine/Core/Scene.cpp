#include "Scene.h"

void Scene::Update()
{
	for (Entity* e : entities)
	{
		e->Update();
	}
}

void Scene::Draw()
{
	for (Entity* e : entities)
	{
		e->Draw();
	}
}

void Scene::Add(Entity* entity)
{
	if (entity != nullptr)
	{
		entities.push_back(entity);
	}
}

Entity* Scene::Remove(Entity* entity, bool destroy)
{
	if (entity != nullptr)
	{
		auto iter = std::find(entities.begin(), entities.end(), entity);
		if (iter != entities.end())
		{
			if (destroy)
			{
				entity->Destroy();
				delete (*iter);
				entity = nullptr;
			}
			entities.erase(iter);
		}
	}

	return entity;
}

Entity* Scene::GetEntityByName(StringId& name)
{
	Entity* entity = nullptr;
	for (Entity* e : entities)
	{
		if (e->GetName() == name)
		{
			entity = e;
		}
	}
	return entity;
}
