#include "Component.h"

#ifndef RENDER_COMPONENT_H
#define RENDER_COMPONENT_H
class RenderComponent : public Component
{
public:
	RenderComponent(StringId& name, Entity* owner) : Component(name, owner) { }

	virtual void Draw() = 0;
	virtual void DrawPick() = 0;

};
#endif