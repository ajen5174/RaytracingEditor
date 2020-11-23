using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace WPFSceneEditor
{
	public static class Engine
	{
		public enum ModelType
        {
			POLYGON_MESH,
			SPHERE
        }
		public enum ComponentType
		{
			NONE = 0,
			TRANSFORM,
			LIGHT,
			MODEL_RENDER,
			CAMERA,
			MATERIAL,
			SPHERE
		}

		public delegate void DebugCallback(string message);
		public delegate void SelectionCallback(float entityID);
		public delegate void SceneLoadedCallback();

		public static string currentFilePath = "";
		public static string outputFilePath = "";

		public static int renderWidth = 266;
		public static int renderHeight = 150;
		public static int maxStringSize = 512;


		[DllImport("user32.dll")]
		public static extern IntPtr SetWindowPos(
									IntPtr handle,
									IntPtr handleAfter,
									int x,
									int y,
									int cx,
									int cy,
									uint flags);
		[DllImport("user32.dll")]
		public static extern IntPtr SetParent(IntPtr child, IntPtr newParent);
		[DllImport("user32.dll")]
		public static extern IntPtr ShowWindow(IntPtr handle, int command);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool ResizeWindow(int width, int height);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool StartEngine();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool InitializeWindow();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern IntPtr GetSDLWindowHandle();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern IntPtr StopEngine();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void RegisterDebugCallback(DebugCallback callback);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void RegisterSelectionCallback(SelectionCallback callback);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void RegisterSceneLoadedCallback(SceneLoadedCallback callback);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool GetFloatData(float entityID, int component, [In, Out] float[] data, int size);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void SetFloatData(float entityID, int component, [In] float[] data, int size);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool GetStringData(float entityID, int component, [In][Out][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] data, int size, int count);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern bool GetIntData(float entityID, int component, [In, Out]int[] data, int size);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void SetStringData(float entityID, int component, [In][Out][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]string[] data, int size, int count);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void ReloadScene(string path);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void SaveScene(string path);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void GetEntityName(float entityID, StringBuilder name);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void GetAllEntityIDs([In, Out] float[] data);
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern int GetEntityCount();

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void EntitySelect(float entityID);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void AddNewEntity();

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void DeleteEntity(float entityID);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void AddComponent(float entityID, int componentType);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void RemoveComponent(float entityID, int componentType);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern float RenameEntity(float entityID, string newName);
	}
}
