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
typedef void(*SelectionCallback) (const float entityID);
typedef void(*SceneLoadedCallback) ();

extern "C" ENGINE_DLL void GetEntityName(float entityID, char* name);
extern "C" ENGINE_DLL void ReloadScene(const char* path);
extern "C" ENGINE_DLL void SaveScene(const char* path);
extern "C" ENGINE_DLL void ResizeWindow(int width, int height);
extern "C" ENGINE_DLL bool StartEngine();
extern "C" ENGINE_DLL HWND GetSDLWindowHandle();
extern "C" ENGINE_DLL bool InitializeWindow();
extern "C" ENGINE_DLL void StopEngine();
extern "C" ENGINE_DLL void RegisterDebugCallback(DebugCallback callback);
extern "C" ENGINE_DLL void RegisterSelectionCallback(SelectionCallback callback);
extern "C" ENGINE_DLL void RegisterSceneLoadedCallback(SceneLoadedCallback callback);
extern "C" ENGINE_DLL bool GetFloatData(float entityID, int component, float* data, int size);
extern "C" ENGINE_DLL void SetFloatData(float entityID, int component, float* data, int size);
extern "C" ENGINE_DLL bool GetStringData(float entityID, int component, char* data[], int size, int count);
extern "C" ENGINE_DLL void SetStringData(float entityID, int component, char* data[], int size, int count);
extern "C" ENGINE_DLL void AddComponent(float entityID, int component);
extern "C" ENGINE_DLL void GetAllEntityIDs(float* data);
extern "C" ENGINE_DLL int GetEntityCount();
extern "C" ENGINE_DLL void EntitySelect(float entityID);
extern "C" ENGINE_DLL void AddNewEntity();


bool InitializeGraphics();
void RunEngine();
void PrintDebugMessage(std::string message);
