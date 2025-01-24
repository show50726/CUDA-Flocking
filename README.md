**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

## CUDA Flocking Simulation
This project is a CUDA-based boids flocking simulation. 
![](images//boid.gif)

It provides 3 different implementations based on different spatial optimizations:
- Naive Implementation
- Uniform Grids
- Coherent Uniform Grids


## Performance Analysis

All the results below were tested on: Windows 10, i7-7700 @ 3.60GHz 16GB, GTX 1080 8GB

### Performance Over Boid Count (No Visualization)
![](images//number_of_boids_to_fps.png)

### Performance Over Boid Count (With Visualization)
![](images//number_of_boids_to_fps(viz).png))

### Performance Over Block Size (No Visualization)
![](images//block_size_to_fps.png)

### Performance over Cell Size (No Visualization)
![](images//block_size_to_fps(viz).png)

