#include "EngineLibrary.h"
#include <string>
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Core/Entity.h"
#include "Renderer/Shader.h"
#include "Renderer/VertexArray.h"
#include "Renderer/Camera.h"
#include "Renderer/Model.h"
#include "Components/ModelRenderComponent.h"
#include "Core/PickTexture.h"
#include "Core/scene.h"

#include "rapidjson/istreamwrapper.h"

#include "Core/Json.h"
#include "Light.h"
#include <time.h>

static SDL_Window* engineWindow = nullptr;
static SDL_GLContext engineContext = nullptr;
static bool isRunning = false;


int windowWidth = 800;
int windowHeight = 600;

bool sizeChanged = false;

#ifdef _WINDLL
const std::string defaultPath = "content\\default_scene.txt";
#else
const std::string defaultPath = "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Scenes\\default.txt";
#endif

std::string gPath = defaultPath;
bool reloadScene = false;

bool quit = false;

Scene* scene;

DebugCallback gDebugCallback;
SelectionCallback gSelectionCallback;
SceneLoadedCallback gSceneLoadedCallback;

ENGINE_DLL void GetBackgroundColor(float* color)
{
	color[0] = scene->backgroundColor.r;
	color[1] = scene->backgroundColor.g;
	color[2] = scene->backgroundColor.b;
}

ENGINE_DLL void SetBackgroundColor(float* color)
{
	scene->backgroundColor = glm::vec3(color[0], color[1], color[2]);
}


ENGINE_DLL int GetEntityCount()
{
	return scene->GetEntities().size();
}


ENGINE_DLL void GetAllEntityIDs(float* data)
{
	auto entities = scene->GetEntities();
	//data = new float[entities.size()];
	for (int i = 0; i < entities.size(); i++)
	{
		data[i] = entities[i]->GetName().GetFloatId();
	}
}


ENGINE_DLL void GetEntityName(float entityID, char* name)
{
	Entity* e = scene->GetEntityByFloatId(entityID);
	if (!e)
	{
		return;
	}
	StringId id = e->GetName();
	memcpy(name, id.cStr(), strlen(id.cStr()) + 1);//add 1 because strlen does NOT include the /0 to terminate the string.
}

ENGINE_DLL void ReloadScene(const char* inPath)
{
	reloadScene = true;
	if(std::strlen(inPath) == 0)
	{
		gPath = defaultPath;
		return;
	}
	gPath = inPath;
}

ENGINE_DLL void SaveScene(const char* outPath)
{
	//save scene here?
	PrintDebugMessage("Saving File...");
	rapidjson::Document json;
	json.SetObject();
	scene->BuildJSON(json);
	auto temp = json.FindMember("entities");

	std::ofstream fstream(outPath);
	rapidjson::OStreamWrapper osw(fstream);
	//FILE* fp = fopen(outPath, "wb");
	char writeBuffer[65536];

	rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
	json.Accept(writer);
	//fclose_s(fp);
	fstream.close();
	PrintDebugMessage("File Saved!");
}

ENGINE_DLL void SetFloatData(float entityID, int component, float* data, int size)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);
	ComponentType type = (ComponentType)component;
	if (!entity)
	{
		return;
	}

	switch (type)
	{
	case ComponentType::NONE:
		break;

	case ComponentType::TRANSFORM:
	{
		Transform* t = entity->GetComponent<Transform>();
		if (size < 9 || !t)
			return;


		t->translation = glm::vec3(data[0], data[1], data[2]);
		//PrintDebugMessage("translation set");

		t->rotation = glm::vec3(glm::radians(data[3]), glm::radians(data[4]), glm::radians(data[5]));

		t->scale = glm::vec3(data[6], data[7], data[8]);

		break;
	}
	case ComponentType::MATERIAL:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (size < 5 || !render)
			return;

		render->model->material->albedo = glm::vec3(data[0], data[1], data[2]);

		render->model->material->fuzz = data[3];
		render->model->material->refractionIndex = data[4];

		break;
	}
	case ComponentType::CAMERA:
	{
		Camera* cam = entity->GetComponent<Camera>();
		if (size < 1 || !cam)
			return;

		cam->fov = data[0];
		cam->SetProjection(cam->fov, cam->aspectRatio, cam->nearClip, cam->farClip);
		break;
	}
	case ComponentType::LIGHT:
	{
		Light* light = entity->GetComponent<Light>();
		if (!light || size < 7)
			return;

		light->color = glm::vec3(data[0], data[1], data[2]);
		light->intensity = data[3];
		light->direction = glm::vec3(data[4], data[5], data[6]);
		break;
	}
	}
}

ENGINE_DLL bool GetStringData(float entityID, int component, char* data[], int size, int count)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);
	ComponentType type = (ComponentType)component;
	if (!entity)
		return false;

	switch (type)
	{
	case ComponentType::NONE:
		break;

	case ComponentType::TRANSFORM:
	{
		break;
	}
	case ComponentType::MODEL_RENDER:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (!render)
			return false;
		if (count < 1 || count > 1)
			return false;
		int filePathSize = render->model->mesh->directory.length() + 1;
		if (filePathSize > size)
			return false;
		data[0] = new char[size];

		memcpy(data[0], render->model->mesh->directory.c_str(), filePathSize);
		break;
	}
	case ComponentType::MATERIAL:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (!render)
			return false;
		if (count < 1 || count > 1)
			return false;
		data[0] = new char[size];
		memcpy(data[0], render->model->material->materialType.c_str(), render->model->material->materialType.length() + 1);

		break;
	}
	case ComponentType::LIGHT:
	{
		Light* light = entity->GetComponent<Light>();
		if (!light)
			return false;
		if (count != 1)
			return false;
		
		data[0] = new char[size];

		memcpy(data[0], light->lightType.c_str(), light->lightType.length() + 1);


		break;
	}
		
	}

	return true;
}

ENGINE_DLL void SetStringData(float entityID, int component, char* data[], int size, int count)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);
	ComponentType type = (ComponentType)component;
	if (!entity)
		return;
	switch (type)
	{
	case ComponentType::NONE:
		break;

	case ComponentType::TRANSFORM:
	{
		break;
	}
	case ComponentType::MODEL_RENDER:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (!render)
			return;
		if (count < 1 || count > 1)
			return;

		PrintDebugMessage("Setting String Data...");
		if (data[0] != render->model->mesh->directory)
		{
			render->model->ReloadMesh(data[0]);
			PrintDebugMessage("Model reloaded");

		}
		else
		{
			render->model->mesh->directory = data[0];
			PrintDebugMessage("Model not reloaded");
		}
		
		break;
	}
	case ComponentType::MATERIAL:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (!render)
			return;
		if (count < 1 || count > 1)
			return;
		render->model->material->materialType = data[0];

		break;
	}
	case ComponentType::LIGHT:
	{
		Light* light = entity->GetComponent<Light>();
		if (!light)
			return;
		if (count != 1)
			return;
		light->lightType = data[0];

		break;
	}
	}

	
}
ENGINE_DLL bool GetIntData(float entityID, int component, int* data, int size)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);

	ComponentType type = (ComponentType)component;
	if (!entity)
		return false;
	switch (type)
	{
	case ComponentType::MODEL_RENDER:
	{
		ModelRenderComponent* m = entity->GetComponent<ModelRenderComponent>();
		if (size > 1 || size < 1 || !m)
			return false;
		data[0] = (int)m->model->modelType;

			break;
	}
	}
}


ENGINE_DLL bool GetFloatData(float entityID, int component, float* data, int size)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);
	ComponentType type = (ComponentType)component;
	if (!entity)
		return false;
	switch (type)
	{
	case ComponentType::NONE:
		break;

	case ComponentType::TRANSFORM:
	{
		Transform* t = entity->GetComponent<Transform>();
		if (size < 9 || !t)
			return false;
		data[0] = t->translation.x;
		data[1] = t->translation.y;
		data[2] = t->translation.z;

		glm::vec3 rotation = glm::eulerAngles(t->rotation) * 180.0f / AI_MATH_PI_F;
		data[3] = rotation.x;
		data[4] = rotation.y;
		data[5] = rotation.z;

		//PrintDebugMessage(std::to_string(rotation.z));

		data[6] = t->scale.x;
		data[7] = t->scale.y;
		data[8] = t->scale.z;
		break;
	}
	case ComponentType::MATERIAL:
	{
		ModelRenderComponent* render = entity->GetComponent<ModelRenderComponent>();
		if (!render || size < 5)
			return false;

		data[0] = render->model->material->albedo.x;//r
		data[1] = render->model->material->albedo.y;//g
		data[2] = render->model->material->albedo.z;//b

		data[3] = render->model->material->fuzz;//fuzz
		data[4] = render->model->material->refractionIndex;//refraction index
		break;
	}
	case ComponentType::CAMERA:
	{
		Camera* cam = entity->GetComponent<Camera>();
		if (!cam || size < 1)
			return false;

		data[0] = cam->fov;
		break;
	}
	case ComponentType::LIGHT:
	{
		Light* light = entity->GetComponent<Light>();
		if (!light || size < 4)
			return false;

		data[0] = light->color.r;
		data[1] = light->color.g;
		data[2] = light->color.b;
		data[3] = light->intensity;
		data[4] = light->direction.x;
		data[5] = light->direction.y;
		data[6] = light->direction.z;
		break;
	}
	}


	return true;
}

ENGINE_DLL void RegisterSceneLoadedCallback(SceneLoadedCallback callback)
{
	if (callback)
	{
		gSceneLoadedCallback = callback;
	}
}

ENGINE_DLL void RegisterSelectionCallback(SelectionCallback callback)
{
	if (callback)
	{
		gSelectionCallback = callback;
	}
}

ENGINE_DLL void RegisterDebugCallback(DebugCallback callback)
{
	if (callback)
	{
		gDebugCallback = callback;
	}
}

ENGINE_DLL void ResizeWindow(int width, int height)
{
	windowWidth = width;
	windowHeight = height;
	sizeChanged = true;
}


ENGINE_DLL void StopEngine()
{
	quit = true;
}

ENGINE_DLL bool StartEngine()
{
    if (isRunning)
        return false;
    isRunning = true;
    RunEngine();

    isRunning = false;
    SDL_GL_DeleteContext(engineContext);
    SDL_DestroyWindow(engineWindow);
    engineContext = nullptr;
    engineWindow = nullptr;
    return true;
}

ENGINE_DLL HWND GetSDLWindowHandle()
{
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);

    SDL_GetWindowWMInfo(engineWindow, &info);
    return info.info.win.window;
}

ENGINE_DLL bool InitializeWindow()
{
    if (engineWindow || engineContext || !InitializeGraphics())
    {
        return false;
    }
}

void PrintDebugMessage(std::string message)
{
	if (gDebugCallback)
	{
		gDebugCallback(message.c_str());
	}
	else
	{
		
	}
}

ENGINE_DLL void EntitySelect(float entityID)
{
	if (gSelectionCallback)
	{
		scene->Deselect();
		if (entityID != 0)
		{
			scene->GetEntityByFloatId(entityID)->selected = true;
		}
		gSelectionCallback(entityID);
		
	}
	else
	{

	}
}

void SceneLoaded()
{
	if (gSceneLoadedCallback)
	{
		gSceneLoadedCallback();
	}
}

ENGINE_DLL void AddNewEntity()
{
	StringId newName = StringId("newEntity", true);
	Entity* entity = new Entity(newName);
	scene->Add(entity);
	SceneLoaded();
}

ENGINE_DLL void DeleteEntity(float entityID)
{
	Entity* e = scene->GetEntityByFloatId(entityID);
	if (!e) return;
	scene->Remove(e);
	SceneLoaded();
}

ENGINE_DLL float RenameEntity(float entityID, char* newName)
{
	Entity* e = scene->GetEntityByFloatId(entityID);
	if (!e) return -1.0f;
	StringId newId = newName;
	if (scene->GetEntityByName(newId))
	{
		PrintDebugMessage("Cannot use duplicate names!");
		return -1.0f;
	}
	e->SetName(newName);
	SceneLoaded();
	return e->GetName().GetFloatId();
}

ENGINE_DLL void CreateRandomSphere()
{
	StringId newName = StringId("sphere", true);
	Entity* entity = new Entity(newName);
	scene->Add(entity);
	AddComponent(entity->GetName().GetFloatId(), (int)ComponentType::SPHERE);

	Transform* transform = entity->GetComponent<Transform>();
	//random transform
	glm::vec3 positionRange = glm::vec3(10.0f);
	glm::vec3 newPosition;
	glm::vec3 scaleRange = glm::vec3(1.0f);
	glm::vec3 newScale;

	newPosition.x = (float)(rand()) / (float)RAND_MAX * (positionRange.x * 2.0f) - positionRange.x;
	newPosition.y = 0.0f;// (float)(rand()) / (float)RAND_MAX * (maxPosition.y * 2.0f) - max.y;
	newPosition.z = (float)(rand()) / (float)RAND_MAX * (positionRange.z * 2.0f) - positionRange.z;

	newScale = glm::vec3((float)(rand()) / (float)RAND_MAX * (scaleRange.x) + 0.5f);
	
	transform->translation = newPosition;
	transform->scale = newScale;


	ModelRenderComponent* mrc = entity->GetComponent<ModelRenderComponent>();
	//randomize material type
	Material* mat = mrc->model->material;
	int randType = rand() % 3;

	switch (randType)
	{
	case 0:
	{
		mat->materialType = "lambert";
		//randomize albedo
		glm::vec3 albedo;
		albedo.x = (float)rand() / RAND_MAX;
		albedo.y = (float)rand() / RAND_MAX;
		albedo.z = (float)rand() / RAND_MAX;
		mat->albedo = albedo;
		break;
	}

	case 1:
	{
		mat->materialType = "metal";
		//randomize albedo and fuzz
		glm::vec3 albedo;
		albedo.x = (float)rand() / RAND_MAX;
		albedo.y = (float)rand() / RAND_MAX;
		albedo.z = (float)rand() / RAND_MAX;
		mat->albedo = albedo;
		mat->fuzz = (float)rand() / RAND_MAX;
		break;
	}

	case 2:
	{
		mat->materialType = "dielectric";
		//randomize refractive index?
		mat->refractionIndex = 1.3;
		break;
	}

	default:
		break;
	}



	SceneLoaded();

}



ENGINE_DLL void AddComponent(float entityID, int component)
{
	Entity* e = scene->GetEntityByFloatId(entityID);
	if (!e) return;
	ComponentType type = (ComponentType)component;
	if (type == ComponentType::MODEL_RENDER)
	{
		ModelRenderComponent* mrc = e->GetComponent<ModelRenderComponent>();
		if (mrc)
		{
			PrintDebugMessage("Already has component");

			return;
		}
		mrc = new ModelRenderComponent(e->GetName(), e);
		e->AddComponent(mrc);
		SceneLoaded();
	}
	else if (type == ComponentType::LIGHT)
	{
		Light* light = e->GetComponent<Light>();
		if (light)
		{
			PrintDebugMessage("Already has component");
			return;
		}
		light = new Light(e->GetName(), e);
		e->AddComponent(light);
		SceneLoaded();
	}
	else if (type == ComponentType::SPHERE)
	{
		ModelRenderComponent* mrc = e->GetComponent<ModelRenderComponent>();
		if (mrc)
		{
			PrintDebugMessage("Already has component");
			return;
		}
		mrc = new ModelRenderComponent(e->GetName(), e);
		mrc->model->modelType = ModelType::SPHERE;
		mrc->model->ReloadMesh("");
		e->AddComponent(mrc);
		SceneLoaded();
	}

}

ENGINE_DLL void RemoveComponent(float entityID, int component)
{
	Entity* e = scene->GetEntityByFloatId(entityID);
	if (!e) return;
	ComponentType type = (ComponentType)component;
	if (type == ComponentType::MODEL_RENDER)
	{
		ModelRenderComponent* mrc = e->GetComponent<ModelRenderComponent>();
		if (mrc)
		{
			e->RemoveComponent(mrc);
			PrintDebugMessage("Component Removed");
			SceneLoaded();
		}
	}
	else if (type == ComponentType::LIGHT)
	{
		Light* light = e->GetComponent<Light>();
		if (light)
		{
			e->RemoveComponent(light);
			PrintDebugMessage("Component Removed");
			SceneLoaded();
		}
	}

}




bool InitializeGraphics()
{
	srand(time(NULL));

	int result = SDL_Init(SDL_INIT_VIDEO);
	if (result != 0)
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
		return false;
	}

#ifdef _WINDLL
	Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE;// | SDL_WINDOW_HIDDEN;
#else
	Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;// | SDL_WINDOW_BORDERLESS;// | SDL_WINDOW_HIDDEN;
#endif

	engineWindow = SDL_CreateWindow("OpenGL", 100, 100, windowWidth, windowHeight, flags); 
	if (engineWindow == nullptr)
	{
		SDL_Log("Failed to create window: %s", SDL_GetError());
		return false;
	}

	//
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetSwapInterval(1);

	engineContext = SDL_GL_CreateContext(engineWindow);
	if (!gladLoadGL()) {
		exit(-1);
	}
	

	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

	//glEnable(GL_CULL_FACE);
	//glFrontFace(GL_CCW);
	//glCullFace(GL_BACK);

	return true;
}

void RunEngine()
{
	
	StringId::AllocNames();

	scene = new Scene();
	
	rapidjson::Document doc;
	if (json::LoadFromFile(gPath, doc))
	{
		scene->Load(doc);
		SceneLoaded();
	}
	else
	{
		PrintDebugMessage("Error reading scene file: " + gPath);
	}


	PickTexture* pick = new PickTexture();
	pick->Initialize(windowWidth, windowHeight);
	
	uint32_t timeA = 0;
	SDL_Event sdlEvent;
	while (!quit)
	{
		uint32_t timeB = SDL_GetTicks();

		uint32_t millisecondsBetween = timeB - timeA;

		if (millisecondsBetween < 16) //60fps for now is fine
		{
			continue;
		}
		else 
		{
			timeA = timeB;
		}

		if (reloadScene)
		{
			reloadScene = false;
			delete scene;
			scene = new Scene();

			rapidjson::Document doc;
			if (json::LoadFromFile(gPath, doc))
			{
				scene->Load(doc);
				SceneLoaded();
			}
			else
			{
				PrintDebugMessage("Error reading scene file");
			}

		}
		//update
		scene->Update();

		//render picking stuff
		pick->EnableWriting();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		scene->DrawPick();
		pick->DisableWriting();

		//draw
		glClearColor(scene->backgroundColor.r, scene->backgroundColor.g, scene->backgroundColor.b, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		scene->Draw();

		SDL_GL_SwapWindow(engineWindow);
		SDL_PollEvent(&sdlEvent);
		switch (sdlEvent.type)
		{
		case SDL_QUIT:
			quit = true;
			break;
		case SDL_KEYDOWN:
			if (sdlEvent.key.keysym.sym == SDLK_ESCAPE)
			{
				quit = true;
			}
			break;
		case SDL_MOUSEBUTTONDOWN:
		{
			int mouseX, mouseY;
			SDL_GetMouseState(&mouseX, &mouseY);
			PickTexture::PickInfo info = pick->ReadPixel(mouseX, windowHeight - mouseY - 1);
			float idRead = info.objectID;
			Entity* selected = scene->GetEntityByFloatId(idRead);
			const char* name;
			//GetEntityName(idRead, name);
			//PrintDebugMessage("ID of suzanne: " + std::to_string(testEntity->GetName().GetId()));
			if (selected)
			{
				//PrintDebugMessage("Selected ID: " +std::to_string(idRead));
				EntitySelect(idRead);
				//selected->selected = true;
			}
			else
			{
				EntitySelect(0);
				//PrintDebugMessage("failed to find ID: " + std::to_string(idRead));
				//SaveScene("C:/Users/Student/OneDrive - Neumont College of Computer Science/Q9 FALL 2020/Capstone Project/CapstoneWork/Source/Content/Scenes/save.txt");
			}
		}
			break;
		case SDL_WINDOWEVENT:
			if (sdlEvent.window.event == SDL_WINDOWEVENT_RESIZED)
			{
				sizeChanged = false;
				PrintDebugMessage("Resizing to " + std::to_string(windowWidth) + " by " + std::to_string(windowHeight));
				//windowWidth = sdlEvent.window.data1;
				//windowHeight = sdlEvent.window.data2;
				SDL_SetWindowSize(engineWindow, windowWidth, windowHeight);
				glViewport(0, 0, windowWidth, windowHeight);
				pick->Resize(windowWidth, windowHeight);
				
			}
			
			break;
		}
		

		SDL_PumpEvents();
	}

	delete scene;
	StringId::FreeNames();
}
