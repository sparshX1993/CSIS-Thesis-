# HD Map Generator

Main Script to generate HD maps sync with GPS and Radar Timestamps.

**What do you need?** 

1) GPS H5 file to get the Longtitude and Latitude and the timestamps
2) Radar H5 file to get the timestamps coordinates
3) Change the paths to the output folder(Save/Rotate/Crop )


**What does the current script do ?**

1) Generates the HD maps based on the latitude and longitude of the h5 file.
2) Take the orientation of the vehicle and align the generated maps to the yaw orientation of the vehicle.
3) Save/Crop/Rotation of the maps to generate the size of 160x160.
4) Converts the images to numpy and then convert the numpy array to the H5 format and save the file.
