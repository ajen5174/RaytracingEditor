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
float vertices[] = {
	-0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	-0.5f,  0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,

	-0.5f, -0.5f,  0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,
	-0.5f, -0.5f,  0.5f,

	-0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,
	-0.5f, -0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,

	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,

	-0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f, -0.5f,
	 0.5f, -0.5f,  0.5f,
	 0.5f, -0.5f,  0.5f,
	-0.5f, -0.5f,  0.5f,
	-0.5f, -0.5f, -0.5f,

	-0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f, -0.5f,
	 0.5f,  0.5f,  0.5f,
	 0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f,  0.5f,
	-0.5f,  0.5f, -0.5f
};

//float vertices[] = {
//	 0.5f,  0.5f, 0.0f,
//	 0.5f, -0.5f, 0.0f,
//	-0.5f,  0.5f, 0.0f,
//
//	 0.5f, -0.5f, 0.0f,
//	-0.5f, -0.5f, 0.0f,
//	-0.5f,  0.5f, 0.0f
//};

float color1[] = { 0.2f, 0.3f, 0.3f };

static SDL_Window* engineWindow = nullptr;
static SDL_GLContext engineContext = nullptr;
static bool isRunning = false;

int windowWidth = 800;
int windowHeight = 600;
bool sizeChanged = false;

bool quit = false;

DebugCallback gDebugCallback;


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
	

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


	//glEnable(GL_CULL_FACE);
	//glFrontFace(GL_CCW);
	//glCullFace(GL_BACK);

	return true;
}

void RunEngine()
{
	StringId::AllocNames();
	StringId entityName = "TestingEntity";
	Entity* testEntity = new Entity(entityName);
	StringId transformName = "TestingTransform";
	Transform* testTransform = new Transform(transformName, testEntity);
	testTransform->translation = glm::vec3(0.0f, 0.0f, -10.0f);
	testEntity->AddComponent(testTransform);

	PrintDebugMessage(testEntity->GetComponent<Transform>()->ToString());

	StringId programName = "BasicShader";
	Shader* program = new Shader(programName);
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

	program->CreateFromFile(vertexPath, GL_VERTEX_SHADER);
	program->CreateFromFile(fragPath, GL_FRAGMENT_SHADER);
	program->Link();
	program->Use();


	VertexArray* va;
	va = new VertexArray();
	va->CreateBuffer(VertexArray::eAttrib::POSITION, sizeof(vertices), 36, vertices);
	va->Bind();
	va->SetAttribute(VertexArray::eAttrib::POSITION, 3, 3 * sizeof(float), 0);

	

	glm::mat4 trans = glm::mat4(1.0f);



	Camera* cam;
	StringId cameraName = "testCamera";
	cam = new Camera(cameraName, testEntity);

	cam->SetProjection(45.0f, (float)windowWidth / (float)windowHeight, 0.01f, 100.0f);
	PrintDebugMessage(std::to_string(windowWidth) + " " + std::to_string(windowHeight));
	testEntity->AddComponent(cam);

	testEntity->Update();


	SDL_Event sdlEvent;
	while (!quit)
	{
		//update
		
		
		//trans = cam->projectionMatrix * glm::rotate(trans, glm::radians(0.05f), glm::vec3(0.0, 0.0, 1.0));
		trans = glm::rotate(trans, glm::radians(0.05f), glm::vec3(0.0f, 0.0f, 1.0f));
		
		
		program->SetUniform("model", trans);
		program->SetUniform("view", cam->viewMatrix);
		program->SetUniform("projection", cam->projectionMatrix);


		//draw
		glClearColor(color1[0], color1[1], color1[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		program->Use();
		va->Draw();

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
			//color1[1] = 1.0f;
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

	delete va;
	delete testEntity;
	delete program;
	StringId::FreeNames();
}
