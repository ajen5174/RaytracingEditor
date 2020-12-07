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
			OutputImageWidthBox.Text = Engine.previousRenderWidth.ToString();
			OutputImageHeightBox.Text = Engine.previousRenderHeight.ToString();
			RecursionDepthBox.Text = Engine.previousRenderDepth.ToString();
			SamplesPerPixelBox.Text = Engine.previousRenderSamples.ToString();
			OutputPathBox.Text = Engine.outputFilePath;
		}

		private void Render_Click(object sender, RoutedEventArgs e)
		{
			//process nonsense
			Process p = new Process();
			p.StartInfo.FileName = "..\\..\\..\\..\\..\\\\Raytracer\\x64\\Release\\Raytracer.exe";

			int samplesPerPixel;
			if (!int.TryParse(SamplesPerPixelBox.Text, out samplesPerPixel) || samplesPerPixel < 1) return;

			int maxRecursionDepth;
			if (!int.TryParse(RecursionDepthBox.Text, out maxRecursionDepth) || maxRecursionDepth < 2) return;

			int outputWidth;
			if (!int.TryParse(OutputImageWidthBox.Text, out outputWidth) || outputWidth < 1) return;

			int outputHeight;
			if (!int.TryParse(OutputImageHeightBox.Text, out outputHeight) || outputHeight < 1) return;

			Engine.previousRenderWidth = outputWidth;
			Engine.previousRenderHeight = outputHeight;
			Engine.previousRenderSamples = samplesPerPixel;
			Engine.previousRenderDepth = maxRecursionDepth;

			p.StartInfo.Arguments = "\"" + Engine.currentFilePath + "\" \"" + Engine.outputFilePath + "\" " + samplesPerPixel + " " + maxRecursionDepth + " " + outputWidth + " " + outputHeight;
			p.Start();

			Close();
		}

		private void OutputFile_Click(object sender, RoutedEventArgs e)
		{
			SaveFileDialog sfd = new SaveFileDialog();
			sfd.Filter = "Image files (*.ppm, *.jpg, *.jpeg, *.png, *.bmp)|*.ppm;*.jpg;*.jpeg;*.png;*.bmp";
			if (sfd.ShowDialog() == true)
			{
				string filename = sfd.FileName;
				Engine.outputFilePath = filename;
				OutputPathBox.Text = filename;
			}
		}
	}
}
