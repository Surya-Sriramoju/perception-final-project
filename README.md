# Semantic Segmentation Using Transformers #
## Segformers architecture has been implement from scratch in pytorch ##

### Dependencies ###
* torch
* matplotlib
* einops
* albumentations
* cityscapesscripts
* opencv
* torchmetrics
* pillow

### steps to train and perform inference ###
* Enter into segformers directory
* add the path to the dataset in the main file, for the variable root_dir
* Uncomment the training code snippet and run the main.py file
* The weights will be stored in a folder called weights
* For inference testing, comment the training snippet and uncomment the testing snippet.
* Provide the path to the weights and run the main.py file

### steps to perform inference using raspberry pi ###
##### note: both raspberry pi and laptop should be in the same network which work on master/slave configuration, both should have the same version of ros #####
* In the src folder, inside utils, provide the path to the weights.
* build the package
* make sure roscore is running on the master, and images are being published by the slave.
* On the master system, execute rosrun cam_sub_node cam_sub_file.py

###### Link to the test inference ######
* https://drive.google.com/file/d/1gtWto1aAiyGkXPuQ824MwXi0A5vE3G_i/view?usp=share_link

##### Note: if you just want to run the inference, please contact the authors for weight file, as github does not allow large files to be uploaded #####

##### Authors #####
* Sai Surya Sriramoju - saisurya@umd.edu
* Dhruv Sharma - dhruvsh@umd.edu
* Mayank Sharma - smayank@umd.edu


