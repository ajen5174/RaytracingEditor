#include "ModelRenderComponent.h"
#include "../Core/Scene.h"
#include "../EngineLibrary.h"


ModelRenderComponent::ModelRenderComponent(StringId& name, Entity* owner)
	: RenderComponent(name, owner)
{
	model = new Model(name.cStr());

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

	pickShader->Use();
	pickShader->SetUniform("mvp", mvpMatrix);
}

void ModelRenderComponent::Draw()
{
	//use shader
	shader->Use();
	//if (!owner->selected)
	{
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
