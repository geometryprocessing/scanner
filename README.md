# scanner
Python Library for 3D Scanning


## TODO:
### Tests
- Write tests for intersection scripts
- Write tests for calibration utils
- Write tests for camera class
- Write tests for projector class
### Camera
- Implement Camera Model enumeration to allow different distortion coefficients 

### Projector
- Complete calibration code for projector
- Test in ipynb environment

### Calibration utils
- Implement get_open_cv_calibration_flags() given a camera model

### Reconstruction Folder
- Folder will be divided into stereoscopy (OpenCV), structured light, structure from motion (COLMAP) gaussian splatting, nerf, and our LookUp method
- Find NERF library and write wrapper Class code for it
- Find Gaussian Splatting library and write wrapper Class code for it

### Github Workflow
- Set workflow to test code after every commit

### PyPi
- Set project to be released in pip