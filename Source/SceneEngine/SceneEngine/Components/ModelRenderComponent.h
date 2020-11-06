#include "RenderComponent.h"
#include "../Renderer/Model.h"
#include "../Renderer/Shader.h"

#ifndef MODEL_RENDER_COMPONENT_H
#define MODEL_RENDER_COMPONENT_H
class ModelRenderComponent : public RenderComponent
{
public:
	ModelRenderComponent(StringId& name, Entity* owner);
	// Inherited via RenderComponent
	virtual void Update() override;
	virtual void Draw() override;
	virtual void DrawPick() override;

	virtual void Destroy() override;
	virtual void Initialize() override;
	virtual bool Load(const rapidjson::Value&) override;

	virtual void BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem) override;

public:
	Model* model;
private:
	Shader* shader;
	Shader* pickShader;
	Shader* outlineShader;


};



#endif