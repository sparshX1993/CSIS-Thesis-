import os


dir_name = "C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Rotate/GPSRADAR/RotatedMAPS"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".png"):
        os.remove(os.path.join(dir_name ,item))




Sep_path_save ='C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_1/Dataset/RadarGPS/'
test2 = os.listdir(Sep_path_save)
for item in test2:
    if item.endswith(".png"):
        os.remove(os.path.join(Sep_path_save, item))




Crop_save_path = r'C:/Users/bjqb7h/Downloads/Thesis2022/HD Maps_Cropped/GPSRADAR'
test3 = os.listdir(Crop_save_path)
for item in test3:
    if item.endswith(".png"):
        os.remove(os.path.join(Crop_save_path, item))

