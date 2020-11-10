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
	/// Interaction logic for TransformEdit.xaml
	/// </summary>
	public partial class TransformEdit : UserControl
	{
		private float selectedEntityID;


		public TransformEdit()
		{
			InitializeComponent();
		}


		public void LoadData(float entityID)
		{
			selectedEntityID = entityID;
			float[] data = new float[9];
			Engine.GetFloatData(selectedEntityID, (int)Engine.ComponentType.TRANSFORM, data, 9);//1 means transform

			TranslationBoxX.Text = "" + data[0];
			TranslationBoxY.Text = "" + data[1];
			TranslationBoxZ.Text = "" + data[2];

			RotationBoxX.Text = "" + data[3];
			RotationBoxY.Text = "" + data[4];
			RotationBoxZ.Text = "" + data[5];

			ScaleBoxX.Text = "" + data[6];
			ScaleBoxY.Text = "" + data[7];
			ScaleBoxZ.Text = "" + data[8];
		}

		private void TranslationBox_KeyUp(object sender, KeyEventArgs e)
		{
			float translationX;
			if (!float.TryParse(TranslationBoxX.Text, out translationX))
				return;
			float translationY;
			if (!float.TryParse(TranslationBoxY.Text, out translationY))
				return;
			float translationZ;
			if (!float.TryParse(TranslationBoxZ.Text, out translationZ))
				return;

			float rotationX;
			if (!float.TryParse(RotationBoxX.Text, out rotationX))
				return;
			float rotationY;
			if (!float.TryParse(RotationBoxY.Text, out rotationY))
				return;
			float rotationZ;
			if (!float.TryParse(RotationBoxZ.Text, out rotationZ))
				return;

			float scaleX;
			if (!float.TryParse(ScaleBoxX.Text, out scaleX))
				return;
			float scaleY;
			if (!float.TryParse(ScaleBoxY.Text, out scaleY))
				return;
			float scaleZ;
			if (!float.TryParse(ScaleBoxZ.Text, out scaleZ))
				return;

			float[] data = { translationX, translationY, translationZ,
							 rotationX, rotationY, rotationZ,
							 scaleX, scaleY, scaleZ };
			Engine.SetFloatData(selectedEntityID, (int)Engine.ComponentType.TRANSFORM, data, 9);
		}

		private void Reset_Click(object sender, RoutedEventArgs e)
		{
			float[] data = { 0, 0, 0, 0, 0, 0, 1, 1, 1};
			Engine.SetFloatData(selectedEntityID, (int)Engine.ComponentType.TRANSFORM, data, 9);
			LoadData(selectedEntityID);
		}
	}
}
