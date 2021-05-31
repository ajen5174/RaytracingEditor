# RaytracingEditor

This project is a 3d scene editor that allows users to organize scenes, and then render them as still images using a raytracing algorithm.
This project uses CUDA. This means you need the CUDA compiler to build it, and it will only run on NVIDIA graphics cards. If you download this project you should be aware that the scene files are set up with absolute file paths, and as such will not load correctly on your machine. If you wish to test it, create a new scene from the start!

## How to build!
1. Install Visual Studio 2019 - This project may not open correctly with other versions of Visual Studio.
2. Install the CUDA 11.0 dev kit. Other versions likely will not work without modifying the Raytracer project file.
3. Download the project and extract the files.
4. Navigate to the Build folder, and open Build.sln using VS2019.
5. Set the configuration to "Build" and "x64".
6. Build the solution.
7. The files are built to Build/x64. 
8. To run the editor simply run WPFSceneEditor.exe! Alternatively you may run it through Visual Studio after building.

## Examples
Here are some example images and scenes built using this project.
![many_spheres](Source/Content/Desktop%20Outputs/random%20spheres/2000spp.png?raw=true "Spheres")
![cornell_box_with_gun](Source/Content/Desktop%20Outputs/cornell%20box%20with%20gun.png?raw=true "Spheres")
![spheres](Source/Content/Desktop%20Outputs/testing%20spheres.png?raw=true "Spheres")

