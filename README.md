**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

## CUDA Flocking Simulation
This project is a CUDA-based boids flocking simulation. 
![](images//boid.gif)

It provides 3 different implementations based on different spatial optimizations:
- Naive Implementation
- Uniform Grids
- Coherent Uniform Grids

## Boid Simulation Algorithm

In the Boids flocking simulation, particles representing birds or fish
(boids) move around the simulation space according to three rules:

1. cohesion - boids move towards the perceived center of mass of their neighbors
2. separation - boids avoid getting to close to their neighbors
3. alignment - boids generally try to move with the same direction and speed as
their neighbors

These three rules specify a boid's velocity change in a timestep.
At every timestep, a boid thus has to look at each of its neighboring boids
and compute the velocity change contribution from each of the three rules.
Thus, a bare-bones boids implementation has each boid check every other boid in
the simulation.

Here are the pseudocode of the 3 rules above:

#### Rule 1: Boids try to fly towards the centre of mass of neighbouring boids

```
function rule1(Boid boid)

    Vector perceived_center

    foreach Boid b:
        if b != boid and distance(b, boid) < rule1Distance then
            perceived_center += b.position
        endif
    end

    perceived_center /= number_of_neighbors

    return (perceived_center - boid.position) * rule1Scale
end
```

#### Rule 2: Boids try to keep a small distance away from other objects (including other boids).

```
function rule2(Boid boid)

    Vector c = 0

    foreach Boid b
        if b != boid and distance(b, boid) < rule2Distance then
            c -= (b.position - boid.position)
        endif
    end

    return c * rule2Scale
end
```

#### Rule 3: Boids try to match velocity with near boids.

```
function rule3(Boid boid)

    Vector perceived_velocity

    foreach Boid b
        if b != boid and distance(b, boid) < rule3Distance then
            perceived_velocity += b.velocity
        endif
    end

    perceived_velocity /= number_of_neighbors

    return perceived_velocity * rule3Scale
end
```

(This section is from `INSTRUCTION.md`)

## Performance Analysis

All the results below were tested on: Windows 10, i7-7700 @ 3.60GHz 16GB, GTX 1080 8GB

### Performance Over Boid Count (No Visualization)
![](images//number_of_boids_to_fps.png)

### Performance Over Boid Count (With Visualization)
![](images//number_of_boids_to_fps(viz).png))

Generally, as we normally expected, the FPS decreases as the boid number raises, since when there're more boids in the scene, there will be more computations to be done. At the same time, more warps need to be dispatch and calculate. Also note that when we disable the visualization, the FPS will be slightly improved since we can get rid of drawing cost.

To be more specific, we can see as the boid number increases, FPS of all the methods decrease. Since the naive method has the O(n<sup>2</sup>) complexity, the FPS decreases significantly as the boid number increases. For the coherent grid method, because we sort the `pos` and `vel` array before accessing, we can further improve the performance by reducing the cache miss (more sequential access to the memory). So we get the best performance for the coherent grid method.

### Performance Over Block Size (No Visualization)
![](images//block_size_to_fps.png)

Note: The boid number is fixed at 50000

### Performance over Block Size (With Visualization)
![](images//block_size_to_fps(viz).png)

Note: The boid number is fixed at 50000

For the block size changes, there's no significant impact on the performance. Overall, block size = 512 seems to have a slightly greater performance than other sizes, and I would guess it is the sweet point of the utilization of the single SM usage and the GPU efficient occupancy and scheduling.

### Performance over Checking Different Number of Neighboring Cells
![](images//27vs8.png)

Note: This is measured under
- coherent grid method
- no visualization

When the number of boids is smaller, the 27 neighbors and 8 neighbors method give a similar FPS result. And as the boid count grows, the 27 neighbors method gives a better performance, especially when the boid number is at about 40000. I believe it is caused by the spatial locality, since we are using the coherent grid method, we sort the array by the grid index, in this case, when we have more grid cells, we're checking the boids more sequenctially. This leads to the better cache usage.
