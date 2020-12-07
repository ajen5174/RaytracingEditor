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
	/// Interaction logic for LightEdit.xaml
	/// </summary>
	public partial class LightEdit : UserControl
	{
		private float entityID;
		private GridLength cachedDirectionRowHeight = new GridLength(0);
		public LightEdit()
		{
			InitializeComponent();
			cachedDirectionRowHeight = DirectionRow.Height;
		}

		public bool LoadData(float entityID)
		{
			this.entityID = entityID;

			float[] data = new float[7];
			if(Engine.GetFloatData(entityID, (int)Engine.ComponentType.LIGHT, data, 7))
			{

				ColorBoxR.Text = "" + data[0];
				ColorBoxG.Text = "" + data[1];
				ColorBoxB.Text = "" + data[2];
				IntensityBox.Text = "" + data[3];
				DirectionBoxX.Text = "" + data[4];
				DirectionBoxY.Text = "" + data[5];
				DirectionBoxZ.Text = "" + data[6];

				string[] stringData = new string[1];
				if(Engine.GetStringData(entityID, (int)Engine.ComponentType.LIGHT, stringData, Engine.maxStringSize, 1))
                {
					StringBuilder sb = new StringBuilder(stringData[0]);
					if (sb.Length > 0) sb[0] = char.ToUpper(sb[0]);
					LightTypeBox.Text = sb.ToString();
					DirectionRowUpdate();
					return true;
				}
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
			float x;
			if (!float.TryParse(DirectionBoxX.Text, out x))
				return;
			float y;
			if (!float.TryParse(DirectionBoxY.Text, out y))
				return;
			float z;
			if (!float.TryParse(DirectionBoxZ.Text, out z))
				return;

			float[] data = { r, g, b, intensity, x, y, z };

			Engine.SetFloatData(entityID, (int)Engine.ComponentType.LIGHT, data, 7);
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
			DirectionRowUpdate();
		}

		private void DirectionRowUpdate()
        {
			if (LightTypeBox.Text == "Point")
			{
				DirectionRow.Height = new GridLength(0);
			}
			else if (LightTypeBox.Text == "Direction")
			{
				DirectionRow.Height = cachedDirectionRowHeight;
			}
			else if (LightTypeBox.Text == "Spotlight")
			{
				DirectionRow.Height = cachedDirectionRowHeight;
			}
		}
		private void ColorBox_KeyUp(object sender, KeyEventArgs e)
		{
			SetFloatData();
		}

		private void DirectionBox_KeyUp(object sender, KeyEventArgs e)
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

		private void LightTypeBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
		{

		}
	}
}
