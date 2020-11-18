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
   
    public partial class ModelRenderEdit : UserControl
    {
        private float selectedEntityID;
        private string meshFilePath;

        public ModelRenderEdit()
        {
            InitializeComponent();
        }

        public bool LoadData(float entityID)
        {
            selectedEntityID = entityID;
            int[] intData = new int[1];
            if (Engine.GetIntData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, intData, 1))
            {
                if (intData[0] == (int)Engine.ModelType.SPHERE)
                    return false;
            }
            string[] stringData = new string[1];

            if (Engine.GetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, stringData, Engine.maxStringSize, 1))
            {
                
                MeshPathBox.Text = stringData[0];
                meshFilePath = stringData[0];
                return true;
            }
            return false;
        }

        private void SelectMesh_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog open = new OpenFileDialog();
            if (open.ShowDialog() == true)
            {
                meshFilePath = open.FileName;
                string[] data = new string[1];
                data[0] = meshFilePath;
                Engine.SetStringData(selectedEntityID, (int)Engine.ComponentType.MODEL_RENDER, data, Engine.maxStringSize, 1);

            }
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
