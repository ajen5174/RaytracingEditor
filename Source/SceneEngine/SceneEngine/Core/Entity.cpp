#include "Entity.h"
#include "../Components/ModelRenderComponent.h"
#include "Json.h"
#include "../Renderer/Camera.h"


void Entity::Update()
{
	//transform might need some fancy stuff here for parent stuff

	for (Component* c : components)
	{
		c->Update();
	}

	for (Entity* child : children)
	{
		child->Update();
	}

}

void Entity::Draw()
{
	RenderComponent* renderComponent = GetComponent<RenderComponent>();
	if (renderComponent)
	{
		renderComponent->Draw();
	}

	for (Entity* child : children)
	{
		child->Draw();
	}

}

void Entity::DrawPick()
{
	RenderComponent* renderComponent = GetComponent<RenderComponent>();
	if (renderComponent)
	{
		renderComponent->DrawPick();
	}

	for (Entity* child : children)
	{
		child->DrawPick();
	}
}

void Entity::AddComponent(Component* component)
{
	if (component != nullptr)
	{
		component->SetOwner(this);
		components.push_back(component);
	}
}

void Entity::RemoveComponent(Component* component)
{
	if (component != nullptr)
	{
		auto iter = std::find(components.begin(), components.end(), component);
		if (iter != components.end())
		{
			delete (*iter);
			components.erase(iter);
		}
	}
}

std::vector<Entity*> Entity::GetChildren()
{
	return children;
}

int Entity::GetFloatData(uint32_t flags, float* data, int size)
{
	//loop through components here to return the right one?
	return 0;
}

void Entity::Destroy()
{
	for (Component* c : components)
	{
		c->Destroy();
		delete c;
	}
	components.clear();
}

bool Entity::Load(const rapidjson::Value& value)
{

	const rapidjson::Value& componentValue = value["components"];
	if (componentValue.IsArray())
	{
		return LoadComponents(componentValue);
	}

	return false;
}

void Entity::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildName(v, "name", name, mem);
	//rapidjson::Value nameKey("name");
	//rapidjson::Value nameValue;
	//nameValue.SetString(name.cStr(), strlen(name.cStr()));
	//v.AddMember(nameKey, nameValue, mem);
	rapidjson::Value componentsValue;
	componentsValue.SetArray();
	
	for (Component* c : components)
	{
		rapidjson::Value tempComponent;
		tempComponent.SetObject();
		c->BuildJSON(tempComponent, mem);
		componentsValue.PushBack(tempComponent, mem);
	}
	rapidjson::Value componentsKey("components");
	v.AddMember(componentsKey, componentsValue, mem);
}


void Entity::Initialize()
{

}

bool Entity::LoadComponents(const rapidjson::Value& value)
{
	for (int i = 0; i < value.Size(); i++)
	{
		const rapidjson::Value& componentValue = value[i];
		std::string componentType;

		json::GetString(componentValue, "type", componentType);

		if (componentType == "Transform")
		{
			StringId temp = name.ToString() + "Transform";
			Transform* transform = new Transform(temp, this);

			if(transform->Load(componentValue))
				AddComponent(transform);
		}
		else if (componentType == "ModelRender")
		{
			StringId rcName = (name.ToString() + "ModelRender");
			ModelRenderComponent* rc = new ModelRenderComponent(rcName, this);
			if(rc->Load(componentValue))
				AddComponent(rc);
		}
		else if (componentType == "Camera")
		{
			StringId camName = name.ToString() + "Camera";
			Camera* cam = new Camera(camName, this);
			if (cam->Load(componentValue))
				AddComponent(cam);
		}
	}
	

	return true;
}
