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

float color1[] = { 0.2f, 0.3f, 0.3f };

static SDL_Window* engineWindow = nullptr;
static SDL_GLContext engineContext = nullptr;
static bool isRunning = false;

int windowWidth = 800;
int windowHeight = 600;
bool sizeChanged = false;

bool quit = false;

Scene* scene;

DebugCallback gDebugCallback;
SelectionCallback gSelectionCallback;


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
		if (size < 9)
			return;

		Transform* t = entity->GetComponent<Transform>();

		t->translation = glm::vec3(data[0], data[1], data[2]);
		//PrintDebugMessage("translation set");

		t->rotation = glm::vec3(glm::radians(data[3]), glm::radians(data[4]), glm::radians(data[5]));

		t->scale = glm::vec3(data[6], data[7], data[8]);

		break;
	}
}


ENGINE_DLL void GetFloatData(float entityID, int component, float* data, int size)
{
	Entity* entity = scene->GetEntityByFloatId(entityID);
	ComponentType type = (ComponentType)component;

	switch (type)
	{
	case ComponentType::NONE:
		break;

	case ComponentType::TRANSFORM:
		Transform* t = entity->GetComponent<Transform>();
		if (size < 9)
			return;
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

void EntitySelect(float entityID)
{
	if (gSelectionCallback)
	{
		gSelectionCallback(entityID);
	}
	else
	{

	}
}

bool InitializeGraphics()
{
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

	std::string path = "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Content\\Scenes\\scene.txt";
	rapidjson::Document doc;
	if (json::LoadFromFile(path, doc))
	{
		scene->Load(doc);
	}
	else
	{
		PrintDebugMessage("Error reading scene file");
	}


	StringId transformName = "TestingTransform";


	StringId camEntityName = "CamEntity";
	Entity* camEntity = new Entity(camEntityName);
	Transform* camTransform = new Transform(transformName, camEntity);
	camTransform->translation = glm::vec3(0.0f, 0.0f, 5.0f);
	camTransform->rotation = glm::vec3(0.0f, glm::radians(180.0f), 0.0f);
	camEntity->AddComponent(camTransform);

	Camera* cam;
	StringId cameraName = "testCamera";
	cam = new Camera(cameraName, camEntity);

	cam->SetProjection(45.0f, (float)windowWidth / (float)windowHeight, 0.01f, 100.0f);
	PrintDebugMessage(std::to_string(windowWidth) + " " + std::to_string(windowHeight));
	camEntity->AddComponent(cam);

	//glm::mat4 trans = glm::mat4(1.0f);
	//trans = glm::rotate(trans, glm::radians(0.05f), glm::vec3(0.0f, 0.0f, 1.0f));

	//scene->Add(testEntity);
	scene->Add(camEntity);
	scene->mainCamera = cam;



	PickTexture* pick = new PickTexture();
	pick->Initialize(windowHeight, windowWidth);
	
	

	SDL_Event sdlEvent;
	while (!quit)
	{
		//update
		scene->Update();
		//trans = cam->projectionMatrix * glm::rotate(trans, glm::radians(0.05f), glm::vec3(0.0, 0.0, 1.0));

		//render picking stuff
		pick->EnableWriting();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		scene->DrawPick();
		pick->DisableWriting();

		//draw
		glClearColor(color1[0], color1[1], color1[2], 1.0f);
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
			//color1[1] = 1.0f;
			//loop through entities to deselect, then check for selection
			//testEntity->selected = false;
			//viewport space
			int mouseX, mouseY;
			SDL_GetMouseState(&mouseX, &mouseY);
			//PrintDebugMessage("Mouse at: " + std::to_string(mouseX) + ", " + std::to_string(mouseY));
			PickTexture::PickInfo info = pick->ReadPixel(mouseX, windowHeight - mouseY - 1);
			//PrintDebugMessage("ID read: " + std::to_string(info.objectID) + ", " + std::to_string(info.primitiveID));
			float idRead = info.objectID;
			Entity* selected = scene->GetEntityByFloatId(idRead);
			//PrintDebugMessage("ID of suzanne: " + std::to_string(testEntity->GetName().GetId()));
			if (selected)
			{
				PrintDebugMessage("Selected ID: " +std::to_string(idRead));
				EntitySelect(idRead);
				selected->selected = true;
			}
			else
			{
				scene->Deselect();
				EntitySelect(0);
				PrintDebugMessage("failed to find ID: " + std::to_string(idRead));
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
				
			}
			
			break;
		}
		

		SDL_PumpEvents();
	}

	delete scene;
	//delete camEntity;
	//delete testEntity;
	StringId::FreeNames();
}
