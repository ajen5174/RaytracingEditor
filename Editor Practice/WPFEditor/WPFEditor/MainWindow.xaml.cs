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

namespace WPFEditor
{
	/// <summary>
	/// Interaction logic for MainWindow.xaml
	/// </summary>
	public partial class MainWindow : Window
	{
		public MainWindow()
		{
			InitializeComponent();
		}


		private void LaunchGraphics(object sender, RoutedEventArgs e)
		{
			Trace.WriteLine("Launching...");
			try
			{
				using (Process myProcess = new Process())
				{
					myProcess.StartInfo.UseShellExecute = false;
					string dir = Directory.GetCurrentDirectory();
					dir = Directory.GetParent(dir).FullName;
					dir = Directory.GetParent(dir).FullName;
					dir = Directory.GetParent(dir).FullName;
					dir = Directory.GetParent(dir).FullName;
					dir = Directory.GetParent(dir).FullName;
					dir += "\\ExampleEngine\\x64\\Release\\ExampleEngine.exe";
					Trace.WriteLine(dir);
					myProcess.StartInfo.FileName = dir;
					myProcess.Start();

				}
			} 
			catch (Exception except)
			{
				Trace.WriteLine(except.Message);
			}
			
		}
	}
}
