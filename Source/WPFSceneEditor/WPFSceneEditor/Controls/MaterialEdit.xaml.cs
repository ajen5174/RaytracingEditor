﻿using Microsoft.Win32;
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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WPFSceneEditor.Controls
{
	/// <summary>
	/// Interaction logic for MaterialEdit.xaml
	/// </summary>
	public partial class MaterialEdit : UserControl
	{
		private float selectedEntityID;
		private GridLength cachedFuzzRowHeight = new GridLength(0);
		private GridLength cachedRefractionRowHeight = new GridLength(0);
		private GridLength cachedAlbedoRowHeight = new GridLength(0);


		public MaterialEdit()
		{
			InitializeComponent();
		}

		public bool LoadData(float entityID)
		{
			cachedFuzzRowHeight = FuzzRow.Height;
			cachedRefractionRowHeight = RefractionRow.Height;
			cachedAlbedoRowHeight = AlbedoRow.Height;

			selectedEntityID = entityID;
			//we need one string, and 5 floats
			string[] stringData = new string[1];

			if (Engine.GetStringData(selectedEntityID, (int)Engine.ComponentType.MATERIAL, stringData, Engine.maxStringSize, 1))
			{
				StringBuilder sb = new StringBuilder(stringData[0]);
				if(sb.Length > 0) sb[0] = char.ToUpper(sb[0]);
				MaterialTypeBox.Text = sb.ToString();
			}

			float[] data = new float[5];
			if(Engine.GetFloatData(selectedEntityID, (int)Engine.ComponentType.MATERIAL, data, 5))
			{
				AlbedoBoxR.Text = "" + data[0];
				AlbedoBoxG.Text = "" + data[1];
				AlbedoBoxB.Text = "" + data[2];

				FuzzBox.Text = "" + data[3];
				RefractionIndex.Text = "" + data[4];

				RowUpdate();
				return true;
			}

			return false;
		}

		private void AlbedoBox_KeyUp(object sender, KeyEventArgs e)
		{
			SetMaterialFloatData();
		}

		private void RowUpdate()
        {
			if(MaterialTypeBox.Text == "Lambert")
            {
				FuzzRow.Height = new GridLength(0);
				RefractionRow.Height = new GridLength(0);
				AlbedoRow.Height = cachedAlbedoRowHeight;
			}
			else if(MaterialTypeBox.Text == "Metal")
            {
				FuzzRow.Height = cachedFuzzRowHeight;
				AlbedoRow.Height = cachedAlbedoRowHeight;
				RefractionRow.Height = new GridLength(0);
			}
			else if(MaterialTypeBox.Text == "Dielectric")
            {
				FuzzRow.Height = new GridLength(0);
				AlbedoRow.Height = new GridLength(0);
				RefractionRow.Height = cachedRefractionRowHeight;

			}
		}


		private void SetMaterialFloatData()
		{
			float albedoR;
			if (!float.TryParse(AlbedoBoxR.Text, out albedoR))
				return;
			float albedoG;
			if (!float.TryParse(AlbedoBoxG.Text, out albedoG))
				return;
			float albedoB;
			if (!float.TryParse(AlbedoBoxB.Text, out albedoB))
				return;

			float fuzz;
			if (!float.TryParse(FuzzBox.Text, out fuzz))
				return;

			float index;
			if (!float.TryParse(RefractionIndex.Text, out index))
				return;

			float[] data = { albedoR, albedoG, albedoB, fuzz, index };
			Engine.SetFloatData(selectedEntityID, (int)Engine.ComponentType.MATERIAL, data, 5);
		}

		private void SetMaterialStringData()
		{
			string[] data = new string[1];
			StringBuilder sb = new StringBuilder(MaterialTypeBox.Text);
			sb[0] = char.ToLower(sb[0]);
			//data[0] = meshFilePath;
			data[0] = sb.ToString();
			Engine.SetStringData(selectedEntityID, (int)Engine.ComponentType.MATERIAL, data, Engine.maxStringSize, 1);
		}

		private void MaterialTypeBox_DropDownClosed(object sender, EventArgs e)
		{
			SetMaterialStringData();
			RowUpdate();

		}


		private void Reset_Click(object sender, RoutedEventArgs e)
		{
			MaterialTypeBox.Text = "Lambert";
			AlbedoBoxR.Text = "" + 0.8f;
			AlbedoBoxG.Text = "" + 0.8f;
			AlbedoBoxB.Text = "" + 0.8f;
			FuzzBox.Text = "" + 0.0f;
			RefractionIndex.Text = "" + 0.0f;
			SetMaterialStringData();
			SetMaterialFloatData();
		}
	}
}
