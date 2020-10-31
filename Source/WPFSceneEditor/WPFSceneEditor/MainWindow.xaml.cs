using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using WPFSceneEditor.Controls;

namespace WPFSceneEditor
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{

		private IntPtr EditorHandle;
		private IntPtr SceneHandle;
		private float EngineAspectRatio = 16f / 9f;
		private float selectedEntityID = 0.0f;

		public MainWindow()
		{
			InitializeComponent();

			var menuDropAlignmentField = typeof(SystemParameters).GetField("_menuDropAlignment", BindingFlags.NonPublic | BindingFlags.Static);
			Action setAlignmentValue = () => {
				if (SystemParameters.MenuDropAlignment && menuDropAlignmentField != null) menuDropAlignmentField.SetValue(null, false);
			};
			setAlignmentValue();
			SystemParameters.StaticPropertyChanged += (sender, e) => { setAlignmentValue(); };


		}

		private void Window_Loaded(object sender, RoutedEventArgs e)
		{
			Dispatcher.BeginInvoke(new Action(() => //this is just here to delay this code until the program is idling.
			{
				Trace.WriteLine("DONE!", "Rendering");
				Trace.WriteLine("Launching...");
				if (Engine.InitializeWindow())
				{
					SceneHandle = Engine.GetSDLWindowHandle();
					EditorHandle = new WindowInteropHelper(this).Handle;

					Engine.RegisterDebugCallback(new Engine.DebugCallback(PrintDebugMessage));
					Engine.RegisterSelectionCallback(new Engine.SelectionCallback(EntitySelect));
					Engine.RegisterSceneLoadedCallback(new Engine.SceneLoadedCallback(SceneLoaded));

					ResizeSceneWindow();

					Trace.WriteLine("Ready to start");

					Engine.StartEngine();

					Trace.WriteLine("Engine closed");

				}

			}), DispatcherPriority.ContextIdle, null); //context idle is the time we are waiting for, after the gui has displayed
			
		}

		private void Button_Click(object sender, RoutedEventArgs e)
		{
			
		}

		private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			Trace.WriteLine("Window closing");
			Engine.StopEngine();
			Trace.WriteLine("Engine stopped");
		}

		

		private void Window_SizeChanged(object sender, SizeChangedEventArgs e)
		{
			ResizeSceneWindow();
		}

		private void ResizeSceneWindow()
		{
			//Trace.WriteLine("Scene Handle: " + SceneHandle.ToString());
			//Trace.WriteLine(EngineContainer.ActualHeight + " " + EngineContainer.ActualWidth);
			Point EngineTopLeft = EngineContainer.TransformToAncestor(this).Transform(new Point(0, 0));
			int newWidth = (int)Math.Ceiling(EngineContainer.ActualWidth);
			int newHeight = (int)((float)Math.Ceiling(EngineContainer.ActualWidth) / EngineAspectRatio);
			if(newHeight > EngineContainer.ActualHeight)
			{
				newHeight = (int)Math.Ceiling(EngineContainer.ActualHeight);
				newWidth = (int)((float)Math.Ceiling(EngineContainer.ActualHeight) * EngineAspectRatio);
			}
			
			Engine.ResizeWindow(newWidth, newHeight);
			Engine.SetWindowPos(SceneHandle, IntPtr.Zero, (int)EngineTopLeft.X, (int)EngineTopLeft.Y, newWidth, newHeight, 0x0040);
			Engine.SetParent(SceneHandle, EditorHandle);
			Engine.ShowWindow(SceneHandle, 1);
		}

		private void SceneLoaded()
		{
			//get all id's of entitys as a float array
			int numEntities = Engine.GetEntityCount();
			float[] ids = new float[numEntities];
			Engine.GetAllEntityIDs(ids);
			
			//loop through that array creating the user controls as we go
			for(int i = 0; i < ids.Count(); i++)
			{
				HierarchyEntity e = new HierarchyEntity();
				e.entityID = ids[i];//assign the id

				//use the id to grab the string name from the engine
				StringBuilder sb = new StringBuilder(256);
				Engine.GetEntityName(e.entityID, sb);
				e.EntityName.Content = sb.ToString().Trim();
				SceneHierarchy.Children.Add(e);
			}
			
		}

		private void PrintDebugMessage(string message)
		{
			Trace.WriteLine(message, "ENGINE_DEBUG");
			DebugOutput.Text += "ENGINE_DEBUG: " + message + "\n";
		}

		public void EntitySelect(float entityID)
		{
			if (selectedEntityID == entityID)
				return;

			selectedEntityID = entityID;
			ComponentEditor.Children.Clear();
			if (entityID == 0)
			{
				return;
			}

			//first we grab the list of components (probably as bitflags)

			//then for each component we add section to add that type of component editing to the window.


			//transform
			TransformEdit te = new TransformEdit();
			te.LoadData(selectedEntityID);
			ComponentEditor.Children.Add(te);

		}

		

		private void Open_Click(object sender, RoutedEventArgs e)
		{
			OpenFileDialog openFile = new OpenFileDialog();
			openFile.Filter = "JSON text files (*.txt;*.json)|*.txt;*.json";
			if(openFile.ShowDialog() == true)
			{
				SceneHierarchy.Children.Clear();
				string path = openFile.FileName;
				Engine.currentFilePath= path;
				Engine.ReloadScene(Engine.currentFilePath);

			}


		}

		private void Save_Click(object sender, RoutedEventArgs e)
		{
			if(Engine.currentFilePath.Length > 1)
			{
				Engine.SaveScene(Engine.currentFilePath);
			}
			else
			{
				SaveAs_Click(sender, e);
			}
			
		}

		private void SaveAs_Click(object sender, RoutedEventArgs e)
		{
			SaveFileDialog save = new SaveFileDialog();
			save.Filter = "JSON text files (*.txt;*.json)|*.txt;*.json";
			if (save.ShowDialog() == true)
			{
				//call engine function here, passing the path to save?
				Engine.currentFilePath = save.FileName;
				Engine.SaveScene(save.FileName);
			}
		}

		private void Render_Click(object sender, RoutedEventArgs e)
		{
			RenderWindow rw = new RenderWindow();

			rw.ShowDialog();
		}
	}
}
