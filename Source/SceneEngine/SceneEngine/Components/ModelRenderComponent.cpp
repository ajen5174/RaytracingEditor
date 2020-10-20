#include "ModelRenderComponent.h"
#include "../Core/Scene.h"


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
}

void ModelRenderComponent::Update()
{
	Transform* trans = owner->GetComponent<Transform>();
	Scene* scene = owner->GetScene();

	//Camera* cam = scene->GetMainCamera();
	Camera* cam = owner->GetComponent<Camera>();

	glm::mat4 modelViewMatrix = cam->viewMatrix * trans->GetMatrix();
	glm::mat4 mvpMatrix = cam->projectionMatrix * modelViewMatrix;

	shader->SetUniform("model", trans->GetMatrix());
	shader->SetUniform("view", cam->viewMatrix);
	shader->SetUniform("projection", cam->projectionMatrix);
}

void ModelRenderComponent::Draw()
{
	//use shader
	shader->Use();
	model->Draw();
}
