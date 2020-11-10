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
	/// Interaction logic for HierarchyEntity.xaml
	/// </summary>
	public partial class HierarchyEntity : UserControl
	{
		public float entityID;
		public Brush defaultBrush;

		public HierarchyEntity()
		{
			InitializeComponent();
			defaultBrush = EntityName.Background;
		}

		private void EntityName_Click(object sender, RoutedEventArgs e)
		{
			Engine.EntitySelect(entityID);
			
		}

		private void Delete_Click(object sender, RoutedEventArgs e)
		{
			Engine.DeleteEntity(entityID);
		}

		private void Rename_Click(object sender, RoutedEventArgs e)
		{

			TextBox tb = new TextBox();
			tb.Name = "RenameTextBox";
			tb.Text = EntityName.Content.ToString();
			ContainerGrid.Children.Add(tb);
			tb.Focus();
			tb.KeyUp += Rename_Submit;

		}

		private void Rename_Submit(object sender, KeyEventArgs e)
		{
			if (e.Key != Key.Enter)
				return;
			TextBox tb = null;
			for(int i = 0; i < ContainerGrid.Children.Count; i++)
			{
				tb = ContainerGrid.Children[i] as TextBox;
				if (tb != null)
					break;
			}
			//TextBox tb = (TextBox)FindName("RenameTextBox");
			EntityName.Content = tb.Text;
			float temp = Engine.RenameEntity(entityID, tb.Text);
			entityID = temp > 0.0f ? temp : entityID;
			ContainerGrid.Children.Remove(tb);
		}

	}
}
