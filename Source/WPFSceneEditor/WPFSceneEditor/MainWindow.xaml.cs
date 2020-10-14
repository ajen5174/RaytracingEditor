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
					
					ResizeSceneWindow();

					Engine.StartEngine();

					Trace.WriteLine("Engine closed");

				}

			}), DispatcherPriority.ContextIdle, null); //context idle is the time we are waiting for, after the gui has displayed
			
		}

		private void Button_Click(object sender, RoutedEventArgs e)
		{
			IntPtr testing = Engine.GetSDLWindowHandle();
			Trace.WriteLine(testing.ToString());
		}

		private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			Trace.WriteLine("Window closing");
			Engine.StopEngine();
			Trace.WriteLine("Engine stopped");
		}

		private static void PrintDebugMessage(string message)
		{
			Trace.WriteLine(message, "ENGINE_DEBUG");
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
			Engine.ResizeWindow((int)Math.Ceiling(EngineContainer.ActualWidth), (int)Math.Ceiling(EngineContainer.ActualHeight));
			Engine.SetWindowPos(SceneHandle, IntPtr.Zero, (int)EngineTopLeft.X, (int)EngineTopLeft.Y, (int)Math.Ceiling(EngineContainer.ActualWidth), (int)Math.Ceiling(EngineContainer.ActualHeight), 0x0040);
			Engine.SetParent(SceneHandle, EditorHandle);
			Engine.ShowWindow(SceneHandle, 1);
		}
	}
}
