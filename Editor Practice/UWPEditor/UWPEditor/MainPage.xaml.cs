using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace UWPEditor
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {

        [DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
        static extern bool StartEngine();
        [DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
        static extern bool InitializeWindow();
        [DllImport("..\\..\\..\\..\\..\\ExampleEngine\\x64\\Release\\ExampleEngine.dll")]
        static extern IntPtr GetSDLWindowHandle();

        public MainPage()
        {
            this.InitializeComponent();

            
        }

        private void LaunchGraphics(object sender, RoutedEventArgs e)
        {
            Console.WriteLine("Launching...");
            InitializeWindow();
            IntPtr windowHandle = GetSDLWindowHandle();
            Console.WriteLine(windowHandle.ToString());
            StartEngine();
        }
    }
}
