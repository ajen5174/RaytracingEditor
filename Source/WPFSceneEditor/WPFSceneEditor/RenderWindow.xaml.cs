using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace WPFSceneEditor
{
	/// <summary>
	/// Interaction logic for RenderWindow.xaml
	/// </summary>
	public partial class RenderWindow : Window
	{
		public RenderWindow()
		{
			InitializeComponent();
		}

		private void Render_Click(object sender, RoutedEventArgs e)
		{
			//process nonsense
			Process p = new Process();
			p.StartInfo.FileName = "C:\\Users\\Student\\OneDrive - Neumont College of Computer Science\\Q9 FALL 2020\\Capstone Project\\CapstoneWork\\Source\\Raytracer\\x64\\Debug\\Raytracer.exe";
			p.StartInfo.Arguments = Engine.currentFilePath + " " + Engine.outputFilePath;
			p.Start();
		}

		private void OutputFile_Click(object sender, RoutedEventArgs e)
		{
			SaveFileDialog sfd = new SaveFileDialog();
			sfd.Filter = "PPM image file (*.ppm)|*.ppm";
			if (sfd.ShowDialog() == true)
			{
				string filename = sfd.FileName;
				Engine.outputFilePath = filename;
				OutputPathBox.Text = filename;
			}
		}
	}
}
