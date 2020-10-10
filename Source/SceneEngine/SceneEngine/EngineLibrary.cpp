#include "EngineLibrary.h"

float vertices[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.0f,  0.5f, 0.0f
};

float color1[] = { 0.2f, 0.3f, 0.3f };

static SDL_Window* engineWindow = nullptr;
static SDL_GLContext engineContext = nullptr;
static bool isRunning = false;


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

bool InitializeGraphics()
{
	int result = SDL_Init(SDL_INIT_VIDEO);
	if (result != 0)
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
		return false;
	}

	Uint32 flags = SDL_WINDOW_OPENGL;// | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN;
	flags |= (false) ? SDL_WINDOW_FULLSCREEN : 0;

	engineWindow = SDL_CreateWindow("OpenGL", 100, 100, 800, 600, flags); 
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
	glCullFace(GL_FRONT);

	return true;
}

void RunEngine()
{
	bool quit = false;
	SDL_Event sdlEvent;
	while (!quit)
	{
		color1[0] = color1[0] > 1.0f ? 0.0f : color1[0] + 0.00001f;
		if (false) //use this here to swap to the other color based on the argument passed in
		{
			glClearColor(0.85f, 0.85f, 0.85f, 1.0f);
		}
		else
		{
			glClearColor(color1[0], color1[1], color1[2], 1.0f);
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
			color1[1] = 1.0f;
			break;
		}

		SDL_PumpEvents();
	}
}
