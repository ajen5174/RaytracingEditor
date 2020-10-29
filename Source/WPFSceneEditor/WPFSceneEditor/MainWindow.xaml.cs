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
			selectedEntityID = entityID;
			if (entityID == 0)
				return;

			float[] data = new float[9];
			Engine.GetFloatData(selectedEntityID, 1, data, 9);

			TranslationBoxX.Text = "" + data[0];
			TranslationBoxY.Text = "" + data[1];
			TranslationBoxZ.Text = "" + data[2];

			RotationBoxX.Text = "" + data[3];
			RotationBoxY.Text = "" + data[4];
			RotationBoxZ.Text = "" + data[5];

			ScaleBoxX.Text = "" + data[6];
			ScaleBoxY.Text = "" + data[7];
			ScaleBoxZ.Text = "" + data[8];

		}

		private void TranslationBox_KeyUp(object sender, KeyEventArgs e)
		{
			float translationX;
			if (!float.TryParse(TranslationBoxX.Text, out translationX))
				return;
			float translationY;
			if (!float.TryParse(TranslationBoxY.Text, out translationY))
				return;
			float translationZ;
			if (!float.TryParse(TranslationBoxZ.Text, out translationZ))
				return;

			float rotationX;
			if (!float.TryParse(RotationBoxX.Text, out rotationX))
				return;
			float rotationY;
			if (!float.TryParse(RotationBoxY.Text, out rotationY))
				return;
			float rotationZ;
			if (!float.TryParse(RotationBoxZ.Text, out rotationZ))
				return;

			float scaleX;
			if (!float.TryParse(ScaleBoxX.Text, out scaleX))
				return;
			float scaleY;
			if (!float.TryParse(ScaleBoxY.Text, out scaleY))
				return;
			float scaleZ;
			if (!float.TryParse(ScaleBoxZ.Text, out scaleZ))
				return;


			float[] data = { translationX, translationY, translationZ,
							 rotationX, rotationY, rotationZ,
							 scaleX, scaleY, scaleZ };
			Engine.SetFloatData(selectedEntityID, 1, data, 9);
		}

		private void Open_Click(object sender, RoutedEventArgs e)
		{
			OpenFileDialog openFile = new OpenFileDialog();
			openFile.Filter = "JSON text files (*.txt;*.json)|*.txt;*.json";
			if(openFile.ShowDialog() == true)
			{
				SceneHierarchy.Children.Clear();
				string path = openFile.FileName;
				Engine.ReloadScene(path);

			}


		}

		private void Save_Click(object sender, RoutedEventArgs e)
		{
			SaveFileDialog save = new SaveFileDialog();
			if(save.ShowDialog() == true)
			{
				//call engine function here, passing the path to save?
				Engine.SaveScene(save.FileName);
			}
		}
	}
}
