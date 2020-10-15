#include "Entity.h"


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
	renderComponent->Draw();

	for (Entity* child : children)
	{
		child->Draw();
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

bool Entity::Load(const rapidjson::Value&)
{
	return false;
}

void Entity::Initialize()
{

}

bool Entity::LoadComponents(rapidjson::Value& value)
{
	return false;
}
