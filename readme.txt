The distance_elevation.csv file contains the geolocation features of the images. 
	The "img_id" column is the index of the data samples. It is tied to the file name in the "imagery" folder. 
	The "distance" column is the approximity of the building from major water bodies.
	The "elevation" column is the elevation of the building
	The "label" column is the label whether a building is damaged or not.
	"coord_x" and "coord_y" are the coordinates of the building. 

The "imagery" folder contains the images cropped from the original satellite imagery of the Greater Houston area before and after Hurricane Harvey in 2015. The image files are named by the indices, which is referenced in the CSV file through the "image_id" column.
