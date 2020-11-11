#include "ModelRenderComponent.h"
#include "../Core/Scene.h"
#include "../EngineLibrary.h"
#include "../Light.h"


ModelRenderComponent::ModelRenderComponent(StringId& name, Entity* owner)
	: RenderComponent(name, owner)
{
	StringId modelName = (name.ToString() + "Model");
	model = new Model(modelName);

	StringId programName = "BasicShader";
	shader = new Shader(programName);
	//read file nonsense?
#ifdef _WINDLL
	std::string vertexPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\matrix_vertex.vert");
#else
	std::string vertexPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\matrix_vertex.vert");
#endif

	char* fragSource;
#ifdef _WINDLL
	std::string fragPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\matrix_fragment.frag");
#else
	std::string fragPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\matrix_fragment.frag");
#endif

	shader->CreateFromFile(vertexPath, GL_VERTEX_SHADER);
	shader->CreateFromFile(fragPath, GL_FRAGMENT_SHADER);
	shader->Link();
	shader->Use();


	char* pickVertexSource;
#ifdef _WINDLL
	std::string pickVertexPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\picking.vert");
#else
	std::string pickVertexPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\picking.vert");
#endif

	char* pickFragSource;
#ifdef _WINDLL
	std::string pickFragPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\picking.frag");
#else
	std::string pickFragPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\picking.frag");
#endif


	StringId pickName = "PickShader";
	pickShader = new Shader(pickName);
	pickShader->CreateFromFile(pickVertexPath, GL_VERTEX_SHADER);
	pickShader->CreateFromFile(pickFragPath, GL_FRAGMENT_SHADER);
	pickShader->Link();
	pickShader->Use();
	pickShader->SetUniform("objectID", owner->GetName().GetFloatId());


	char* outlineVertexSource;
#ifdef _WINDLL
	std::string outlineVertexPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\outline.vert");
#else
	std::string outlineVertexPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\outline.vert");
#endif

	char* outlineFragSource;
#ifdef _WINDLL
	std::string outlineFragPath("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\outline.frag");
#else
	std::string outlineFragPath("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\outline.frag");
#endif


	StringId outlineName = "OutlineShader";
	outlineShader = new Shader(outlineName);
	outlineShader->CreateFromFile(outlineVertexPath, GL_VERTEX_SHADER);
	outlineShader->CreateFromFile(outlineFragPath, GL_FRAGMENT_SHADER);
	outlineShader->Link();
	outlineShader->Use();

}

void ModelRenderComponent::Update()
{
	Transform* trans = owner->GetComponent<Transform>();
	Scene* scene = owner->GetScene();

	Camera* cam = scene->GetMainCamera();
	//Camera* cam = owner->GetComponent<Camera>();

	glm::mat4 modelViewMatrix = cam->viewMatrix * trans->GetMatrix();
	glm::mat4 mvpMatrix = cam->projectionMatrix * cam->viewMatrix * trans->GetMatrix();

	shader->Use();
	shader->SetUniform("mvp", mvpMatrix);
	shader->SetUniform("mv", modelViewMatrix);

	auto lights = scene->Get<Light>();
	for (int i = 0; i < (*lights).size(); i++)
	{
		(*lights)[i]->SetShader("lights[" + std::to_string(i) + "]", shader);
	}

	shader->SetUniform("lightPosition", glm::vec3(2.0f, 2.0f, 2.0f));

	pickShader->Use();
	pickShader->SetUniform("mvp", mvpMatrix);
	pickShader->SetUniform("mv", modelViewMatrix);
}

void ModelRenderComponent::Draw()
{
	

	if (owner->selected)
	{
		//stencil set up
		glEnable(GL_STENCIL_TEST);
		glEnable(GL_DEPTH_TEST);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); 
		glStencilMask(0x00);

		// Render the mesh into the stencil buffer.
		glStencilFunc(GL_ALWAYS, 1, 0xFF);
		glStencilMask(0xFF);
		//use the normal shader. This renders to the normal AND stencil buffer.
		model->material->Use(shader);
		//	shader->Use();
		model->Draw();

		// Render the same mesh with a shader that will scale it out
			//use the stencil here to prevent writing to any pixel that has the model we drew in it.
		glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
		glStencilMask(0x00);
		//glDisable(GL_DEPTH_TEST); // uncomment this for a different way of highlightin, it is more obvious but looks weirder I think.
		//use the outline shader
		outlineShader->Use();
		Scene* scene = owner->GetScene();
		Camera* cam = scene->GetMainCamera();
		glm::mat4 mvpMatrix = cam->projectionMatrix * cam->viewMatrix * owner->GetComponent<Transform>()->GetMatrix();
		outlineShader->SetUniform("mvp", mvpMatrix);
		//draw
		model->Draw();
		//reset stencil nonsense
		glStencilMask(0xFF);
		glStencilFunc(GL_ALWAYS, 1, 0xFF);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_STENCIL_TEST);


	}
	else
	{
		model->material->Use(shader);
		//shader->Use();
		model->Draw();
	}
	
}

void ModelRenderComponent::DrawPick()
{
	//use shader
	pickShader->Use();
	//PrintDebugMessage(std::to_string(owner->GetName().GetFloatId()));
	//if (!owner->selected)
	{
		model->Draw();
	}

}

bool ModelRenderComponent::Load(const rapidjson::Value& value)
{
	return model->Load(value);
}

void ModelRenderComponent::BuildJSON(rapidjson::Value& v, rapidjson::MemoryPoolAllocator<>& mem)
{
	json::BuildCString(v, "type", "ModelRender", mem);
	model->BuildJSON(v, mem);
}

void ModelRenderComponent::Destroy()
{
}

void ModelRenderComponent::Initialize()
{
}
