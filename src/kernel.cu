#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char* msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3* dev_pos;
glm::vec3* dev_vel1;
glm::vec3* dev_vel2;

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int* dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// additional buffers used to reshuffle the position and velocity 
// data to be coherent within cells.
thrust::device_ptr<glm::vec3> dev_thrust_particlePosIndices;
thrust::device_ptr<glm::vec3> dev_thrust_particleVelIndices;

// Grid parameters based on simulation parameters.
// These are automatically computed in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;
float maxDistance;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* A typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* A basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3* arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	// Don't forget to cudaFree in  Boids::endSimulation.
	cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects,
		dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// Computing grid params
	maxDistance = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	gridCellWidth = 2.0f * maxDistance;
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3* vel, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float* vbodptr_positions, float* vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
	glm::vec3 perceivedCenter = glm::vec3(0, 0, 0);
	glm::vec3 perceivedVel = glm::vec3(0, 0, 0);
	glm::vec3 c = glm::vec3(0, 0, 0);
	int neighborCount1 = 0;
	int neighborCount2 = 0;
	for (int i = 0; i < N; i++) {
		float distance = glm::distance(pos[i], pos[iSelf]);
		// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
		if (iSelf != i && distance <= rule1Distance) {
			neighborCount1++;
			perceivedCenter += pos[i];
		}

		// Rule 2: boids try to stay a distance d away from each other
		if (iSelf != i && distance <= rule2Distance) {
			c -= (pos[i] - pos[iSelf]);
		}

		// Rule 3: boids try to match the speed of surrounding boids
		if (iSelf != i && distance <= rule3Distance) {
			perceivedVel += vel[i];
			neighborCount2++;
		}
	}
	glm::vec3 rule1Vel = glm::vec3(0, 0, 0);
	if (neighborCount1 > 0) {
		perceivedCenter /= neighborCount1;
		rule1Vel = (perceivedCenter - pos[iSelf]) * rule1Scale;
	}

	glm::vec3 rule2Vel = c * rule2Scale;

	glm::vec3 rule3Vel = glm::vec3(0, 0, 0);
	if (neighborCount2 > 0) {
		perceivedVel /= neighborCount2;
		rule3Vel = perceivedVel * rule3Scale;
	}


	return vel[iSelf] + rule1Vel + rule2Vel + rule3Vel;
}

/**
* Basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
	glm::vec3* vel1, glm::vec3* vel2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Compute a new velocity based on pos and vel1
	glm::vec3 newVel = computeVelocityChange(N, index, pos, vel1);

	// Clamp the speed
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// Record the new velocity into vel2. Question: why NOT vel1?
	vel2[index] = newVel;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3* pos, glm::vec3* vel) {
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	thisPos += vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

	thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// Consider this method of computing a 1D index from a 3D grid index.
// Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* indices, int* gridIndices) {
	// - Label each boid with the index of its grid cell.
	// - Set up a parallel array of integer indices as pointers to the actual
	//   boid data in pos and vel1/vel2
	int boidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (boidIndex >= N) {
		return;
	}

	glm::vec3 position = pos[boidIndex];
	glm::ivec3 gridIndex = glm::ivec3(
		imax(0, imin(gridResolution - 1, floor((position.x - gridMin.x) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((position.y - gridMin.y) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((position.z - gridMin.z) * inverseCellWidth)))
	);

	int grid1DIndex = gridIndex3Dto1D(gridIndex.x, gridIndex.y, gridIndex.z, gridResolution);

	indices[boidIndex] = boidIndex;
	gridIndices[boidIndex] = grid1DIndex;
}

// For indicating that a cell does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
	int* gridCellStartIndices, int* gridCellEndIndices) {
	// Identify the start point of each cell in the gridIndices array.
	// This is basically a parallel unrolling of a loop that goes
	// "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) {
		return;
	}

	if (index == 0) {
		gridCellStartIndices[particleGridIndices[0]] = 0;
	}
	if (index == N - 1) {
		gridCellEndIndices[particleGridIndices[N - 1]] = N - 1;
		return;
	}

	int thisGrid = particleGridIndices[index];
	int nextGrid = particleGridIndices[index + 1];
	if (thisGrid != nextGrid) {
		gridCellStartIndices[nextGrid] = index + 1;
		gridCellEndIndices[thisGrid] = index;
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	// - Identify the grid cell that this particle is in
	glm::vec3 boidPosition = pos[index];
	glm::ivec3 gridIndex = glm::ivec3(
		imax(0, imin(gridResolution - 1, floor((boidPosition.x - gridMin.x) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((boidPosition.y - gridMin.y) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((boidPosition.z - gridMin.z) * inverseCellWidth)))
	);

	float halfCellWidth = cellWidth * 0.5f;
	float xDir = 1, yDir = 1, zDir = 1;
	// TODO: Can be optimized
	glm::ivec3 pGridIndex = glm::ivec3(
		imax(0, floor((boidPosition.x - halfCellWidth - gridMin.x) * inverseCellWidth)),
		imax(0, floor((boidPosition.y - halfCellWidth - gridMin.y) * inverseCellWidth)),
		imax(0, floor((boidPosition.z - halfCellWidth - gridMin.z) * inverseCellWidth))
	);
	xDir += (pGridIndex.x - gridIndex.x) * 2;
	yDir += (pGridIndex.y - gridIndex.y) * 2;
	zDir += (pGridIndex.z - gridIndex.z) * 2;

	glm::vec3 perceivedCenter = glm::vec3(0, 0, 0);
	glm::vec3 perceivedVel = glm::vec3(0, 0, 0);
	glm::vec3 c = glm::vec3(0, 0, 0);
	int neighborCount1 = 0;
	int neighborCount2 = 0;
	// - Identify which cells may contain neighbors.
	for (int i = 0; i <= 1; i++) {
		for (int j = 0; j <= 1; j++) {
			for (int k = 0; k <= 1; k++) {
				int gridX = gridIndex.x + i * xDir;
				int gridY = gridIndex.y + j * yDir;
				int gridZ = gridIndex.z + k * zDir;
				if (gridX >= 0 && gridX < gridResolution && gridY >= 0 && gridY < gridResolution && gridZ >= 0 && gridZ < gridResolution) {
					int grid1DIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
					// - For each cell, read the start/end indices in the boid pointer array.
					int startIndex = gridCellStartIndices[grid1DIndex];
					int endIndex = gridCellEndIndices[grid1DIndex];

					if (startIndex > 0) {
						for (int l = startIndex; l <= endIndex; l++) {
							int neighborBoidIndex = particleArrayIndices[l];
							// - Access each boid in the cell and compute velocity change from
							//   the boids rules, if this boid is within the neighborhood distance.
							glm::vec3 neighborBoidPos = pos[neighborBoidIndex];
							glm::vec3 neighborBoidVel = vel1[neighborBoidIndex];
							float distance = glm::distance(neighborBoidPos, boidPosition);
							// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
							if (index != neighborBoidIndex && distance <= rule1Distance) {
								neighborCount1++;
								perceivedCenter += neighborBoidPos;
							}

							// Rule 2: boids try to stay a distance d away from each other
							if (index != neighborBoidIndex && distance <= rule2Distance) {
								c -= (neighborBoidPos - boidPosition);
							}

							// Rule 3: boids try to match the speed of surrounding boids
							if (index != neighborBoidIndex && distance <= rule3Distance) {
								perceivedVel += neighborBoidVel;
								neighborCount2++;
							}
						}
					}
				}
			}
		}
	}
	glm::vec3 rule1Vel = glm::vec3(0, 0, 0);
	if (neighborCount1 > 0) {
		perceivedCenter /= neighborCount1;
		rule1Vel = (perceivedCenter - boidPosition) * rule1Scale;
	}

	glm::vec3 rule2Vel = c * rule2Scale;

	glm::vec3 rule3Vel = glm::vec3(0, 0, 0);
	if (neighborCount2 > 0) {
		perceivedVel /= neighborCount2;
		rule3Vel = perceivedVel * rule3Scale;
	}

	glm::vec3 newVel = vel1[index] + rule1Vel + rule2Vel + rule3Vel;
	// - Clamp the speed change before putting the new speed in vel2
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = newVel * (maxSpeed / speed);
	}

	vel2[index] = newVel;
}

__device__ glm::ivec3 positionTo3DGridIndex(glm::vec3 position, glm::vec3 gridMin,
	int gridResolution, float inverseCellWidth) {
	return glm::ivec3(
		imax(0, imin(gridResolution - 1, floor((position.x - gridMin.x) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((position.y - gridMin.y) * inverseCellWidth))),
		imax(0, imin(gridResolution - 1, floor((position.z - gridMin.z) * inverseCellWidth)))
	);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth, float maxDistance,
	int* gridCellStartIndices, int* gridCellEndIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// This should be very similar to kernUpdateVelNeighborSearchScattered,
	// except with one less level of indirection.
	// This should expect gridCellStartIndices and gridCellEndIndices to refer
	// directly to pos and vel1.
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N)
		return;

	// - Identify the grid cell that this particle is in
	glm::vec3 boidPosition = pos[index];
	glm::ivec3 gridIndex = positionTo3DGridIndex(boidPosition, gridMin,
		gridResolution, inverseCellWidth);

	// - Identify which cells may contain neighbors. This isn't always 8.
	glm::vec3 range = glm::vec3(maxDistance, maxDistance, maxDistance);
	glm::ivec3 minGridIndex = positionTo3DGridIndex(boidPosition - range, gridMin,
		gridResolution, inverseCellWidth);
	glm::ivec3 maxGridIndex = positionTo3DGridIndex(boidPosition + range, gridMin,
		gridResolution, inverseCellWidth);

	glm::vec3 perceivedCenter = glm::vec3(0, 0, 0);
	glm::vec3 perceivedVel = glm::vec3(0, 0, 0);
	glm::vec3 c = glm::vec3(0, 0, 0);
	int neighborCount1 = 0;
	int neighborCount2 = 0;
	//   DIFFERENCE: For best results, consider what order the cells should be
	//   checked in to maximize the memory benefits of reordering the boids data.
	for (int gridZ = minGridIndex.z; gridZ <= maxGridIndex.z; gridZ++) {
		for (int gridY = minGridIndex.y; gridY <= maxGridIndex.y; gridY++) {
			for (int gridX = minGridIndex.x; gridX <= maxGridIndex.x; gridX++) {
				if (gridX >= 0 && gridX < gridResolution && gridY >= 0 && gridY < gridResolution
					&& gridZ >= 0 && gridZ < gridResolution) {
					int grid1DIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
					// - For each cell, read the start/end indices in the boid pointer array.
					int startIndex = gridCellStartIndices[grid1DIndex];
					int endIndex = gridCellEndIndices[grid1DIndex];

					if (startIndex > 0) {
						// - Access each boid in the cell and compute velocity change from
						//   the boids rules, if this boid is within the neighborhood distance.
						for (int l = startIndex; l <= endIndex; l++) {
							glm::vec3 neighborBoidPos = pos[l];
							glm::vec3 neighborBoidVel = vel1[l];
							float distance = glm::distance(neighborBoidPos, boidPosition);
							// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
							if (index != l && distance <= rule1Distance) {
								neighborCount1++;
								perceivedCenter += neighborBoidPos;
							}

							// Rule 2: boids try to stay a distance d away from each other
							if (index != l && distance <= rule2Distance) {
								c -= (neighborBoidPos - boidPosition);
							}

							// Rule 3: boids try to match the speed of surrounding boids
							if (index != l && distance <= rule3Distance) {
								perceivedVel += neighborBoidVel;
								neighborCount2++;
							}
						}
					}
				}
			}
		}
	}
	glm::vec3 rule1Vel = glm::vec3(0, 0, 0);
	if (neighborCount1 > 0) {
		perceivedCenter /= neighborCount1;
		rule1Vel = (perceivedCenter - boidPosition) * rule1Scale;
	}

	glm::vec3 rule2Vel = c * rule2Scale;

	glm::vec3 rule3Vel = glm::vec3(0, 0, 0);
	if (neighborCount2 > 0) {
		perceivedVel /= neighborCount2;
		rule3Vel = perceivedVel * rule3Scale;
	}

	glm::vec3 newVel = vel1[index] + rule1Vel + rule2Vel + rule3Vel;
	// - Clamp the speed change before putting the new speed in vel2
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = newVel * (maxSpeed / speed);
	}

	vel2[index] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// Use the kernels to step the simulation forward in time.
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects,
		dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	// ping-pong the velocity buffers
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	dim3 fullBlocksPerGridBoid((numObjects + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGridCell((gridCellCount + blockSize - 1) / blockSize);

	// - label each particle with its array index as well as its grid index.
	//   Use 2x width grids.
	kernComputeIndices << <fullBlocksPerGridBoid, blockSize >> > (numObjects,
		gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// - Unstable key sort using Thrust.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices);

	// - Reset the start indices buffer and end indicies buffer to -1, indicating no boid inside the cell
	kernResetIntBuffer << <fullBlocksPerGridCell, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer dev_gridCellStartIndices failed!");
	kernResetIntBuffer << <fullBlocksPerGridCell, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer dev_gridCellEndIndices failed!");

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGridBoid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGridBoid, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
	cudaDeviceSynchronize();

	// - Update positions
	kernUpdatePos << <fullBlocksPerGridBoid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	cudaDeviceSynchronize();

	// - Ping-pong buffers as needed
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
	// In Parallel:
	dim3 fullBlocksPerGridBoid((numObjects + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGridCell((gridCellCount + blockSize - 1) / blockSize);

	// - Label each particle with its array index as well as its grid index.
	//   Use 2x width grids
	kernComputeIndices << <fullBlocksPerGridBoid, blockSize >> > (numObjects,
		gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// - Unstable key sort using Thrust.
	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
	dev_thrust_particlePosIndices = thrust::device_ptr<glm::vec3>(dev_pos);
	dev_thrust_particleVelIndices = thrust::device_ptr<glm::vec3>(dev_vel1);
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
		thrust::make_zip_iterator(dev_thrust_particleArrayIndices, dev_thrust_particlePosIndices, dev_thrust_particleVelIndices));

	// - Reset the start indices buffer and end indicies buffer to -1, indicating no boid inside the cell
	kernResetIntBuffer << <fullBlocksPerGridCell, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer dev_gridCellStartIndices failed!");
	kernResetIntBuffer << <fullBlocksPerGridCell, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer dev_gridCellEndIndices failed!");

	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of boid indices
	kernIdentifyCellStartEnd << <fullBlocksPerGridBoid, blockSize >> > (numObjects, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// - Perform velocity updates using neighbor search
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGridBoid, blockSize >> > (numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth, gridCellWidth, maxDistance, dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");
	cudaDeviceSynchronize();

	// - Update positions
	kernUpdatePos << <fullBlocksPerGridBoid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");
	cudaDeviceSynchronize();

	// - Ping-pong buffers as needed.
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);

	// Free additional buffers here.
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
	// test unstable sort
	int* dev_intKeys;
	int* dev_intValues;
	int N = 10;

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// LOOK-2.1 Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}
