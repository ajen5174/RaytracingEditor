#include "Scene.h"
#include "../EngineLibrary.h"
#include "Json.h"
#include "../Components/ModelRenderComponent.h"

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
		if(!e->selected)
			e->Draw();
	}
	for (Entity* e : entities)
	{
		if (e->selected)
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

void Scene::Deselect()
{
	for (Entity* e : entities)
	{
		e->selected = false;
	}
}

void Scene::BuildJSON(rapidjson::Document& doc)
{
	json::BuildInt(doc, "EntityCount", entities.size(), doc.GetAllocator());
	json::BuildVec3(doc, "BackgroundColor", backgroundColor, doc.GetAllocator());
	rapidjson::Value entityArray;
	entityArray.SetArray();
	rapidjson::Value name;
	name.SetString("entities");



	for (Entity* e : entities)
	{
		rapidjson::Value entityValue;
		entityValue.SetObject();
		e->BuildJSON(entityValue, doc.GetAllocator());
		entityArray.PushBack(entityValue, doc.GetAllocator());
	}
	doc.AddMember(name, entityArray, doc.GetAllocator());
}


void Scene::Load(rapidjson::Value& value)
{
	PrintDebugMessage("Loading file...");
	json::GetVec3(value, "BackgroundColor", backgroundColor);

	const rapidjson::Value& enititiesArray = value["entities"];
	if (enititiesArray.IsArray())
	{
		for (rapidjson::SizeType i = 0; i < enititiesArray.Size(); i++)
		{
			const rapidjson::Value& entityValue = enititiesArray[i];
			if (entityValue.IsObject())
			{
				StringId entityName;
				json::GetName(entityValue, "name", entityName);

				//StringId testing = "thisshouldwork";
				Entity* entity = new Entity(entityName);
				entity->scene = this;
				if (entity->Load(entityValue))
				{
					PrintDebugMessage("Entity Added");
					Add(entity);
				}
				else
				{
					PrintDebugMessage("Failed to add entity");
					delete entity;
				}
			}
			else
			{
				PrintDebugMessage("File is not an object" + std::to_string(i));
			}
		}
	}
	else
	{
		PrintDebugMessage("no entities");
	}
	PrintDebugMessage("File loaded!");

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
