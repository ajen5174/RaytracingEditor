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

		[DllImport("user32.dll")]
		private static extern IntPtr SetWindowPos(
									IntPtr handle,
									IntPtr handleAfter,
									int x,
									int y,
									int cx,
									int cy,
									uint flags);
		[DllImport("user32.dll")]
		private static extern IntPtr SetParent(IntPtr child, IntPtr newParent);
		[DllImport("user32.dll")]
		private static extern IntPtr ShowWindow(IntPtr handle, int command);

		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		static extern bool StartEngine();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		static extern bool InitializeWindow();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		static extern IntPtr GetSDLWindowHandle();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		static extern IntPtr StopEngine();
		[DllImport("..\\..\\..\\..\\..\\SceneEngine\\x64\\Release\\SceneEngine.dll")]
		static extern void RegisterDebugCallback(DebugCallback callback);


		private delegate void DebugCallback(string message);

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
				if (InitializeWindow())
				{
					RegisterDebugCallback(new DebugCallback(PrintDebugMessage));
					IntPtr windowHandle = GetSDLWindowHandle();
					Trace.WriteLine(windowHandle.ToString());

					SetWindowPos(windowHandle, IntPtr.Zero, 200, 200, 0, 0, 0x0401);
					IntPtr editorWindow = new WindowInteropHelper(this).Handle;
					SetParent(windowHandle, editorWindow);
					ShowWindow(windowHandle, 1);
					StartEngine();

					Trace.WriteLine("Engine closed");

				}

			}), DispatcherPriority.ContextIdle, null); //context idle is the time we are waiting for, after the gui has displayed
			
		}

		private void Button_Click(object sender, RoutedEventArgs e)
		{
			IntPtr testing = GetSDLWindowHandle();
			Trace.WriteLine(testing.ToString());
		}

		private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			Trace.WriteLine("Window closing");
			StopEngine();
			Trace.WriteLine("Engine stopped");
		}

		private static void PrintDebugMessage(string message)
		{
			Trace.WriteLine(message, "ENGINE_DEBUG");
		}
	}
}
