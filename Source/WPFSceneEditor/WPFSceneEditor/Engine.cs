using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace WPFSceneEditor
{
	public static class Engine
	{
		public delegate void DebugCallback(string message);
		public delegate void SelectionCallback(float entityID);
		public delegate void SceneLoadedCallback();

		public static string currentFilePath = "";
		public static string outputFilePath = "";



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
		public static extern void GetFloatData(float entityID, int component, [In, Out] float[] data, int size);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		public static extern void SetFloatData(float entityID, int component, [In] float[] data, int size);
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

		

	}
}
