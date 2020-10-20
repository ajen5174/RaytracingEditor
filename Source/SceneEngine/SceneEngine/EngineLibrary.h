#define NOMINMAX //this define prevents the windows.h from compiling their min and max functions, which collide with assimp for some reason
#include <SDL.h>
#include <glad/glad.h>
#include <SDL_syswm.h>
#include <string>

#ifdef SCENEENGINE_EXPORTS
#define ENGINE_DLL __declspec(dllexport)
#else
#define ENGINE_DLL __declspec(dllimport)
#endif

typedef void(*DebugCallback) (const char* str);

extern "C" ENGINE_DLL void ResizeWindow(int width, int height);
extern "C" ENGINE_DLL bool StartEngine();
extern "C" ENGINE_DLL HWND GetSDLWindowHandle();
extern "C" ENGINE_DLL bool InitializeWindow();
extern "C" ENGINE_DLL void StopEngine();
extern "C" ENGINE_DLL void RegisterDebugCallback(DebugCallback callback);

bool InitializeGraphics();
void RunEngine();
void PrintDebugMessage(std::string message);
