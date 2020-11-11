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
	/// Interaction logic for CameraEdit.xaml
	/// </summary>
	public partial class CameraEdit : UserControl
	{
		private float entityID;

		public CameraEdit()
		{
			InitializeComponent();
		}

		public bool LoadData(float entityID)
		{
			this.entityID = entityID;

			float[] data = new float[1];
			if(Engine.GetFloatData(entityID, (int)Engine.ComponentType.CAMERA, data, 1))
			{
				FovBox.Text = "" + data[0];
				return true;
			}
			return false;
		}

		private void Reset_Click(object sender, RoutedEventArgs e)
		{
			FovBox.Text = "" + 45.0f;
			SetData();
		}


		private void SetData()
		{
			float fov;
			if(float.TryParse(FovBox.Text, out fov))
			{
				float[] data = { fov };
				Engine.SetFloatData(entityID, (int)Engine.ComponentType.CAMERA, data, 1);
			}
		}


		private void FovBox_KeyUp(object sender, KeyEventArgs e)
		{
			SetData();
		}
	}
}
