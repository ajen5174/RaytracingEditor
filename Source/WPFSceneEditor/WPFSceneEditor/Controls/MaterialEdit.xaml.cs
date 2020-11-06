using Microsoft.Win32;
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
		private string meshFilePath;
		private int maxStringSize = 512;

		public MaterialEdit()
		{
			InitializeComponent();
		}

		public bool LoadData(float entityID)
		{
			selectedEntityID = entityID;
			//we need one string, and 5 floats
			string[] stringData = new string[2];

			if (Engine.GetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, stringData, maxStringSize, 2))
			{
				MeshPathBox.Text = stringData[0];
				meshFilePath = stringData[0];
				//stringData[1][0] = char.ToUpper(stringData[1][0]);
				StringBuilder sb = new StringBuilder(stringData[1]);
				if(sb.Length > 0) sb[0] = char.ToUpper(sb[0]);
				MaterialTypeBox.Text = sb.ToString();
			}

			float[] data = new float[5];
			if(Engine.GetFloatData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, data, 5))
			{
				AlbedoBoxR.Text = "" + data[0];
				AlbedoBoxG.Text = "" + data[1];
				AlbedoBoxB.Text = "" + data[2];

				FuzzBox.Text = "" + data[3];
				RefractionIndex.Text = "" + data[4];

				return true;
			}

			return false;
		}

		private void AlbedoBox_KeyUp(object sender, KeyEventArgs e)
		{
			//need to set the string as well

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

			//string materialType = MaterialTypeBox.Text;
			//materialType = materialType.ToLower();
			//Engine.SetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, materialType);

			float[] data = { albedoR, albedoG, albedoB, fuzz, index };
			Engine.SetFloatData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, data, 5);
		}

		private void SelectMesh_Click(object sender, RoutedEventArgs e)
		{
			OpenFileDialog open = new OpenFileDialog();
			if(open.ShowDialog() == true)
			{
				meshFilePath = open.FileName;
				SetModelStringData();

			}
		}

		private void SetModelStringData()
		{
			string[] data = new string[2];
			StringBuilder sb = new StringBuilder(MaterialTypeBox.Text);
			sb[0] = char.ToLower(sb[0]);
			data[0] = meshFilePath;
			data[1] = sb.ToString();
			Engine.SetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, data, maxStringSize, 2);
		}

		private void MaterialTypeBox_DropDownClosed(object sender, EventArgs e)
		{
			SetModelStringData();

		}
	}
}
