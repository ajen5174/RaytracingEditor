using System;
using System.Collections.Generic;
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
	/// Interaction logic for SettingsWindow.xaml
	/// </summary>
	public partial class SettingsWindow : Window
	{
		public SettingsWindow()
		{
			InitializeComponent();
			BackgroundColorRed.Text = "" + Engine.backgroundColor[0];
			BackgroundColorGreen.Text = "" + Engine.backgroundColor[1];
			BackgroundColorBlue.Text = "" + Engine.backgroundColor[2];
		}


		private void Confirm_Click(object sender, RoutedEventArgs e)
		{
			float red;
			if (!float.TryParse(BackgroundColorRed.Text, out red)) return;
			float green;
			if (!float.TryParse(BackgroundColorGreen.Text, out green)) return;
			float blue;
			if (!float.TryParse(BackgroundColorBlue.Text, out blue)) return;

			Engine.backgroundColor[0] = red;
			Engine.backgroundColor[1] = green;
			Engine.backgroundColor[2] = blue;

			Engine.SetBackgroundColor(Engine.backgroundColor);

			Close();
		}
	}
}
