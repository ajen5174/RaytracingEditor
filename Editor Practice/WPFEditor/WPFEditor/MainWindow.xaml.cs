using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;
using System.Windows.Interop;

namespace WPFEditor
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
									uint flags
		);
		[DllImport("user32.dll")]
		private static extern IntPtr SetParent(IntPtr child, IntPtr newParent);
		[DllImport("user32.dll")]
		private static extern IntPtr ShowWindow(IntPtr handle, int command);

		[DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
		static extern bool StartEngine();
		[DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
		static extern bool InitializeWindow();
		[DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
		static extern IntPtr GetSDLWindowHandle();


		public MainWindow()
		{
			InitializeComponent();

			
			

		}


		private void LaunchGraphics(object sender, RoutedEventArgs e)
		{
			Trace.WriteLine("Launching...");
			if(InitializeWindow())
			{
				IntPtr windowHandle = GetSDLWindowHandle();
				Trace.WriteLine(windowHandle.ToString());

				SetWindowPos(windowHandle, IntPtr.Zero, 200, 200, 0, 0, 0x0401);
				IntPtr editorWindow = new WindowInteropHelper(this).Handle;
				SetParent(windowHandle, editorWindow);
				ShowWindow(windowHandle, 1);
				StartEngine();
			}
			

		}
	}
}
