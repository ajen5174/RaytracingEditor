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
using System.Windows.Controls.Primitives;
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

		//private void ScrollBoxToBottom()
  //      {
		//	DebugOutput.ScrollToEnd();
		//}

		private void Window_Loaded(object sender, RoutedEventArgs e)
		{
			//DebugOutput.TextChanged += (sender, e) =>
			//{
			//	DebugOutput.CaretIndex = DebugOutput.Text.Length;
			//	DebugOutput.ScrollToEnd();
			//};

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
			SceneHierarchy.Children.Clear();
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
			DebugOutput.Text += ("ENGINE_DEBUG: " + message + "\n");
			DebugScrollViewer.ScrollToEnd();
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
			for(int i = 0; i < SceneHierarchy.Children.Count; i++)
			{
				HierarchyEntity he = SceneHierarchy.Children[i] as HierarchyEntity;
				if(he != null)
				{
					if (he.entityID == selectedEntityID)
					{
						he.EntityName.Background = new SolidColorBrush(Color.FromRgb(0x99, 0x99, 0x99));
					}
					else
					{
						he.EntityName.Background = he.defaultBrush;
					}
				}
			}


			//transform
			TransformEdit te = new TransformEdit();
			te.LoadData(selectedEntityID);
			ComponentEditor.Children.Add(te);

			ModelRenderEdit mre = new ModelRenderEdit();
			SphereEdit se = new SphereEdit();
			if(mre.LoadData(selectedEntityID))
			{
				ComponentEditor.Children.Add(mre);

				MaterialEdit me = new MaterialEdit();
				if (me.LoadData(selectedEntityID))
					ComponentEditor.Children.Add(me);
			}
			else if(se.LoadData(selectedEntityID))
            {
				ComponentEditor.Children.Add(se);

				MaterialEdit me = new MaterialEdit();
				if (me.LoadData(selectedEntityID))
					ComponentEditor.Children.Add(me);
			}

			

			CameraEdit ce = new CameraEdit();
			if (ce.LoadData(selectedEntityID))
				ComponentEditor.Children.Add(ce);

			LightEdit le = new LightEdit();
			if (le.LoadData(selectedEntityID))
				ComponentEditor.Children.Add(le);

		}

		

		private void Open()
		{
			OpenFileDialog openFile = new OpenFileDialog();
			openFile.Filter = "JSON text files (*.txt;*.json)|*.txt;*.json";
			if(openFile.ShowDialog() == true)
			{
				string path = openFile.FileName;
				Engine.currentFilePath = path;
				Engine.ReloadScene(Engine.currentFilePath);

			}


		}

		private void Save()
		{
			if(Engine.currentFilePath.Length > 1)
			{
				Engine.SaveScene(Engine.currentFilePath);
			}
			else
			{
				SaveAs();
			}
			
		}

		private void SaveAs()
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
			rw.Owner = this;
			rw.ShowDialog();
		}

		private void AddEntity_Click(object sender, RoutedEventArgs e)
		{
			Engine.AddNewEntity();
		}

		private void AddModelRender_Click(object sender, RoutedEventArgs e)
		{
			Engine.AddComponent(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER);

			float temp = selectedEntityID;
			EntitySelect(0);
			EntitySelect(temp);//reselect to show new component
		}

		private void AddSphere_Click(object sender, RoutedEventArgs e)
        {
			Engine.AddComponent(selectedEntityID, (int)Engine.ComponentType.SPHERE);

			float temp = selectedEntityID;
			EntitySelect(0);
			EntitySelect(temp);//reselect to show new component
		}

		private void New()
		{
			Engine.currentFilePath = "";
			Engine.ReloadScene("");
		}

		private void Exit_Click(object sender, RoutedEventArgs e)
		{
			Application.Current.Shutdown();
		}

		private void AddLight_Click(object sender, RoutedEventArgs e)
		{
			Engine.AddComponent(selectedEntityID, (int)Engine.ComponentType.LIGHT);

			float temp = selectedEntityID;
			EntitySelect(0);
			EntitySelect(temp);//reselect to show new component
		}

		private void NewCommandBinding_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			New();
		}

		private void OpenCommandBinding_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			Open();
		}

		private void SaveCommandBinding_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			Save();
		}

		private void SaveAsCommandBinding_Executed(object sender, ExecutedRoutedEventArgs e)
		{
			SaveAs();
		}

	}
}
