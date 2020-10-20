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

void Scene::DrawPick()
{
	for (Entity* e : entities)
	{
		e->DrawPick();
	}
}

void Scene::Add(Entity* entity)
{
	if (entity != nullptr)
	{
		entity->scene = this;
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

Entity* Scene::GetEntityByName(uint32_t name)
{
	Entity* entity = nullptr;
	for (Entity* e : entities)
	{
		if (e->GetName().GetId() == name)
		{
			entity = e;
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

Entity* Scene::GetEntityByFloatId(float id)
{
	Entity* entity = nullptr;
	for (Entity* e : entities)
	{
		if (e->GetName().GetFloatId() == id)
		{
			entity = e;
		}
	}
	return entity;
}
