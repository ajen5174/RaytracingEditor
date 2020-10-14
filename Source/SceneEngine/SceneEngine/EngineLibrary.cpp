#include "EngineLibrary.h"
#include <string>
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
float vertices[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.0f,  0.5f, 0.0f
};

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


	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);

	return true;
}

void RunEngine()
{
	int success;
	char infoLog[512];

	char* vertexSource;
#ifdef _WINDLL
	std::ifstream stream("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\vertex.vert", std::ios::binary|std::ios::ate);
#else
	std::ifstream stream("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\vertex.vert", std::ios::binary | std::ios::ate);
#endif
	if (stream.is_open())
	{
		int size = stream.tellg();
		vertexSource = new char[size];
		PrintDebugMessage(std::to_string(size));
		stream.seekg(0, std::ios::beg);
		stream.read(vertexSource, size);

		if (vertexSource[size] != '\0')
		{
			PrintDebugMessage("Not null terminated");
		}

		stream.close();
	}
	else
	{
		PrintDebugMessage("Vertex shader could not be opened");
		return;
	}

	

	GLuint vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::VERTEX::SHADER::COMPILE_FAILED\n" << infoLog << std::endl;
		PrintDebugMessage(infoLog);
		quit = true;
	}

	char* fragSource;
#ifdef _WINDLL
	std::ifstream streamTwo("..\\..\\..\\..\\..\\SceneEngine\\SceneEngine\\Shaders\\fragment.frag", std::ios::binary|std::ios::ate);
#else
	std::ifstream streamTwo("C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\SceneEngine\\SceneEngine\\Shaders\\fragment.frag", std::ios::binary|std::ios::ate);
#endif
	if (streamTwo.is_open())
	{
		size_t size = streamTwo.tellg();
		fragSource = new char[size];
		streamTwo.seekg(0, std::ios::beg);
		streamTwo.read(fragSource, size);
		streamTwo.close();
	}
	else
	{
		PrintDebugMessage("Fragment shader could not be opened");
		return;
	}



	GLuint fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::VERTEX::SHADER::COMPILE_FAILED\n" << infoLog << std::endl;
		PrintDebugMessage(infoLog);
		quit = true;
	}

	GLuint shaderProgram;
	shaderProgram = glCreateProgram();

	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		PrintDebugMessage(std::string(infoLog) + "Shader source: \n" + fragSource);
		quit = true;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	glUseProgram(shaderProgram);

	GLuint vbo;
	glGenBuffers(1, &vbo);

	GLuint vao;
	glGenVertexArrays(1, &vao);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glm::mat4 trans = glm::mat4(1.0f);

	SDL_Event sdlEvent;
	while (!quit)
	{
		//update
		trans = glm::rotate(trans, glm::radians(0.05f), glm::vec3(0.0, 0.0, 1.0));
		GLuint transformLoc = glGetUniformLocation(shaderProgram, "transform");
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, (float*)&trans);

		//draw
		glClearColor(color1[0], color1[1], color1[2], 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);

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

	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glDeleteProgram(shaderProgram);
}
