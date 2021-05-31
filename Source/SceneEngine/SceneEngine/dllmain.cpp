// dllmain.cpp : Defines the entry point for the DLL application.

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
#include "EngineLibrary.h"


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

//This main should only be called if we are testing manually by building to an exe. Not used in the final build
int main(int argc, char** argv)
{
    InitializeWindow();
    ReloadScene("C:/Users/Student/OneDrive - Neumont College of Computer Science/Q9 FALL 2020/Capstone Project/CapstoneWork/Source/Content/Scenes/scene.txt");
    StartEngine();
    return 0;
}