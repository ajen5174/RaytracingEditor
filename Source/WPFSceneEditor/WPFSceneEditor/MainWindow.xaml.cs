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

					ResizeSceneWindow();

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

		private void PrintDebugMessage(string message)
		{
			Trace.WriteLine(message, "ENGINE_DEBUG");
			DebugOutput.Text += "ENGINE_DEBUG: " + message + "\n";
		}

		private void EntitySelect(float entityID)
		{
			selectedEntityID = entityID;

			float[] data = new float[9];
			Engine.GetFloatData(selectedEntityID, 1, data, 9);

			TranslationBoxX.Text = "" + data[0];
			TranslationBoxY.Text = "" + data[1];
			TranslationBoxZ.Text = "" + data[2];


			//DebugOutput.Text += "DATA: " + data[5] + "\n";

			//DebugOutput.Text += "ENGINE_SELECT: " + entityID + "\n";

		}

		private void TranslationBox_KeyUp(object sender, KeyEventArgs e)
		{
			float translationX = float.Parse(TranslationBoxX.Text);
			float translationY = float.Parse(TranslationBoxY.Text);
			float translationZ = float.Parse(TranslationBoxZ.Text);
			float[] data = { translationX, translationY, translationZ};
			Engine.SetFloatData(selectedEntityID, 1, data, 3);
		}
	}
}
