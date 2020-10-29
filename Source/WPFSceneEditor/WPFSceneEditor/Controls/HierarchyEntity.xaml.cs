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

		public HierarchyEntity()
		{
			InitializeComponent();
		}

		private void EntityName_Click(object sender, RoutedEventArgs e)
		{
			Engine.EntitySelect(entityID);
		}
	}
}
