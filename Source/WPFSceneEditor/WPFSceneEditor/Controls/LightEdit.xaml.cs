﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WPFSceneEditor.Controls
{
	/// <summary>
	/// Interaction logic for LightEdit.xaml
	/// </summary>
	public partial class LightEdit : UserControl
	{
		private float entityID;

		public LightEdit()
		{
			InitializeComponent();
		}

		public bool LoadData(float entityID)
		{
			this.entityID = entityID;

			float[] data = new float[4];
			if(Engine.GetFloatData(entityID, (int)Engine.ComponentType.LIGHT, data, 4))
			{

				ColorBoxR.Text = "" + data[0];
				ColorBoxG.Text = "" + data[1];
				ColorBoxB.Text = "" + data[2];
				IntensityBox.Text = "" + data[3];
				return true;
			}



			return false;
		}


		private void SetFloatData()
		{
			float r;
			if (!float.TryParse(ColorBoxR.Text, out r))
				return;
			float g;
			if (!float.TryParse(ColorBoxG.Text, out g))
				return;
			float b;
			if (!float.TryParse(ColorBoxB.Text, out b))
				return;
			float intensity;
			if (!float.TryParse(IntensityBox.Text, out intensity))
				return;


			float[] data = { r, g, b, intensity };

			Engine.SetFloatData(entityID, (int)Engine.ComponentType.LIGHT, data, 4);
		}

		private void SetStringData()
		{
			if (LightTypeBox.Text.Length < 1)
				return;
			string[] data = new string[1];
			StringBuilder sb = new StringBuilder(LightTypeBox.Text);
			sb[0] = char.ToLower(sb[0]);
			data[0] = sb.ToString();
			Engine.SetStringData(entityID, (int)Engine.ComponentType.LIGHT, data, Engine.maxStringSize, 1);
		}


		private void ColorBox_KeyUp(object sender, KeyEventArgs e)
		{
			SetFloatData();
		}

		private void LightTypeBox_DropDownClosed(object sender, EventArgs e)
		{
			SetStringData();
		}

		private void Delete_Click(object sender, RoutedEventArgs e)
		{
			Engine.RemoveComponent(entityID, (int)Engine.ComponentType.LIGHT);

			float temp = entityID;
			Engine.EntitySelect(0);
			Engine.EntitySelect(temp);
		}

		private void Reset_Click(object sender, RoutedEventArgs e)
		{
			LightTypeBox.Text = "Point";
			ColorBoxR.Text = "" + 1;
			ColorBoxG.Text = "" + 1;
			ColorBoxB.Text = "" + 1;
			IntensityBox.Text = "" + 1;
			SetFloatData();
			SetStringData();
		}
	}
}
