#include <sdl.h>
#include <glad/glad.h>

float vertices[] = {
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
     0.0f,  0.5f, 0.0f
};

float color1[] = {0.2f, 0.3f, 0.3f};

int main(int argc, char** argv)
{
	int result = SDL_Init(SDL_INIT_VIDEO);
	if (result != 0)
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
		return false;
	}

	Uint32 flags = SDL_WINDOW_OPENGL;
	flags |= (false) ? SDL_WINDOW_FULLSCREEN : 0;

	SDL_Window* window = SDL_CreateWindow("OpenGL", 100, 100, 800, 600, flags);
	if (window == nullptr)
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

	SDL_GLContext context = SDL_GL_CreateContext(window);
	if (!gladLoadGL()) {
		exit(-1);
	}

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_FRONT);


	bool quit = false;
	SDL_Event sdlEvent;
	while (!quit)
	{
		if (false) //use this here to swap to the other color based on the argument passed in
		{
			glClearColor(0.85f, 0.85f, 0.85f, 1.0f);
		}
		else
		{
			glClearColor(color1[0], color1[1], color1[2], 1.0f);
		}
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		SDL_GL_SwapWindow(window);
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
		}
		SDL_PumpEvents();
	}

	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
}