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

    public partial class SphereEdit : UserControl
    {
        private float selectedEntityID;

        public SphereEdit()
        {
            InitializeComponent();
        }

        public bool LoadData(float entityID)
        {
            selectedEntityID = entityID;
            string[] stringData = new string[1];
            if (Engine.GetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, stringData, Engine.maxStringSize, 1))
            {
                return true;

            }
            return false;
        }


        private void Delete_Click(object sender, RoutedEventArgs e)
        {
            Engine.RemoveComponent(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER);

            float temp = selectedEntityID;
            Engine.EntitySelect(0);
            Engine.EntitySelect(temp);
        }
    }
}
