# scanner
Python Library for 3D Scanning


## TODO:
### Tests
- Write tests for calibration utils
- Write tests for camera class
- Write tests for projector class

### Camera
- Implement Camera Model enumeration to allow different distortion coefficients

### Projector

### Calibration utils
- Implement get_open_cv_calibration_flags() given a camera model

### Structured Light

### LookUp
- B Spline fitting requires the knots to be passed -- higher frequency patterns need more knots
- Consider the memory efficient way of handling the calibration of the look up table -- one frame at time makes the most sense, but where to store? what filename? does user need to input? 
- Think about mask generation for the LookUp method -- should the user also be able to input a region of interest tuple to reduce the computational cost?
- Start with simple index fetching (np.argmin(np.linalg.norm(data - table))) and then move onto making it differentiable with weigths

### COLMAP
- Write wrapper Class to make it easier to access the COLMAP API that we need the most for our reconstruction projects
- Need to define what are the things we want with COLMPA -- for example, we want to be able to pass charuco markers from outside, we want to pass intrinsic and extrinsic parameters and see how they compare with the COLMAP's optimization solutions

### Metashape
- Do we want Metashape as part of this library? It is proprietary and requires a license, but Giancarlo has written a lot of the code for the Paleo project and it can easily be recycled into here
- Convert paleo pipeline form dagster pipeline into a class with function calls that are easier than metashape's API

### NERF
- Find NERF library and write wrapper Class code for it

### Gaussian Splatting
- Find Gaussian Splatting library and write wrapper Class code for it

### Github Workflow
- Set workflow to test code after every commit

### PyPi
- Set project to be released in pip