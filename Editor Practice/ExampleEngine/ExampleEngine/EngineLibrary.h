#pragma once
#include <SDL_syswm.h>

#ifdef EXAMPLEENGINEDLL_EXPORTS
#define ENGINE_DLL __declspec(dllexport)
#else
#define ENGINE_DLL __declspec(dllimport)
#endif

extern "C" ENGINE_DLL bool StartEngine();
extern "C" ENGINE_DLL HWND GetSDLWindowHandle();
extern "C" ENGINE_DLL bool InitializeWindow();
bool InitializeSDLAndOpenGL();
void RunEngine();