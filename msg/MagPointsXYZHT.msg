# Time of sensor data acquisition, and the coordinate frame ID
Header header

# 2D structure of the magnetometers. If only one layer of magnetometers are used, then height is 1 and width is the number of the megnetometers.
uint32 height   # layers
uint32 width    # number of magnetometers per layer

mag_msg/MagPointXYZHT[] mag_points