//
//  main.cu
//  TermProject
//
//  Created by Kutay Demireren on 25/12/15.
//  Copyright © 2015 Kutay Demireren. All rights reserved.
//
/*****************************************
This program compute n many particles forces on each other and their
velocities & positions according to forces in time.
It is called N-Body Solver.

This code is written by Kutay Demireren and Elif Ecem Ates. January, 2016.
*****************************************/


#include <iostream>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <fstream>

using namespace std;

/* Dimension of vect arrays */
#define DIM 2

/* Global definitions for X and Y */
#define X 0
#define Y 1

/* Universal Gravity Constant */
#define G 6.673*pow(10,-11)

typedef double vect_t[DIM];

struct particle_t{
    double mass;
    double position[2];
    double velocity[2];
};

/* Function prototypes */
void print_particles(particle_t *particles);
double fRand(double fMin, double fMax);
void init_force(vect_t *vect, int N);
void init_particle(particle_t *vect, int N);
double getTime();
void read_input(particle_t *particles, int N, char *file);
void usage(int argc, char **argv);

int N = 0;
int num_threads = 1;

/* Kernel functions */
__global__ void compute_force_on_device(vect_t *forces, particle_t *particles, int N, double g, double *force_qk);
__global__ void compute_pos_and_vel_on_device(vect_t *forces, particle_t *particles, int delta_t, int N, double g);


int main(int argc, char * argv[]) {
    srand((unsigned)time(0));
    /*
     Inputs to the program(respectively):
     N            number of the particles
     delta_t      time difference between two time
     T            final time number
     debug        1 to see each step, 0 to see only final step.
     num_threads  # of threads working on the program.

     Inputs provided via file:
     (for each particle)
     mass           mass of particle
     initial_pos    initial position of particle
     initial_vel    initial velocities of particle
     */

    int step = 0, T = 0, delta_t, debug, n_steps;
    vect_t *forces;
    particle_t *particles;
    double start, end, total_time;
    char *pfile;

    // Error code to check CUDA return values
    cudaError_t err = cudaSuccess;

    /* check validity of inputs	*/
    if (argc != 7)
        usage(argc, argv);
    if ((N = atoi(argv[1])) <= 0 ||
        (delta_t = atoi(argv[2])) <= 0 ||
        (T = atoi(argv[3])) <= 0 ||
        (debug = atoi(argv[4])) < 0 ||
	      (debug = atoi(argv[4])) > 1 ||
        (num_threads = atoi(argv[5])) <= 0)
        usage(argc, argv);



    /* Allocating on host memory */
    forces = (vect_t *) malloc(N*sizeof(vect_t));
    particles = (particle_t *) malloc(N*sizeof(particle_t));
    double *force_qk = (double *) malloc(2*sizeof(double));
    if(!forces || !particles || !force_qk){
        fprintf(stderr, "error: unable to allocate memory\n");
        exit(1);
    }

    //Initalizing host vectors
    init_force(forces, N);
    pfile = argv[6];
    read_input(particles, N, pfile);


    cout << endl << "Start computing n-body solver" << endl;
    cout << "------------------" << endl;


    //Size of each vector
    size_t size_force = N*sizeof(vect_t);
    size_t size_particle = N*sizeof(particle_t);
    //Initialize device input vectors
    vect_t *d_forces;
    particle_t *d_particles;
    double *d_force_qk;

    /* Allocating memory on device */
    // Allocate the device input vector d_forces
    err = cudaMalloc((void **)&d_forces, size_force);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate device vector d_forces (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

    // Allocate the device input vector d_particles
    err = cudaMalloc((void **)&d_particles, size_particle);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate device vector d_particles (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }


    err = cudaMalloc((void **)&d_force_qk, 2*sizeof(double));
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate device vector d_force_qk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }


    /* Copy host vectors to device vectors */
    // Copy forces to d_forces
    err = cudaMemcpy(d_forces, forces, size_force, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy vector d_forces from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

    // Copy particles to d_particles
    err = cudaMemcpy(d_particles, particles, size_particle, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy vector d_particles from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }

    // Copy particles to d_force_qk
    err = cudaMemcpy(d_force_qk, force_qk, 2*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy vector d_force_qk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }


    //Final setups for CUDA
    int numThreadsInBlock = num_threads;
    int numBlocks = N / numThreadsInBlock + (N%numThreadsInBlock == 0 ? 0 : 1);

    cout << "Computing starts with..." << endl << " Block Size: " << numThreadsInBlock << endl << " # of Blocks : " << numBlocks << endl;

    start = getTime();

    n_steps = T / delta_t;

    double g = G;

    //Computing
    for (step = 0; step < n_steps; step++) {
      /* Synchronize, before it compute new velocities and positions.
	     To make sure all computations have finished in the last step */
      cudaThreadSynchronize();

  if(step % 100 == 0 && debug == 1){
  	if(step == 0)
	   cout << "Initial particles" << endl;
	  else
	   cout << "Iteration " << step << endl;

	/* Take current values to print */
	err = cudaMemcpy(particles, d_particles, size_particle, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	  {
	    fprintf(stderr, "Failed to copy vector particles from device to host (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	  }

	print_particles(particles);
      }

      //Set all forces on particles to 0 before computing
      err = cudaMemset(d_forces, 0, size_force);
      if(err != cudaSuccess){
	fprintf(stderr, "Failed to set the memory to value 0 (error code %s)!\n", cudaGetErrorString(err));
	exit(1);
      }

      //Kernel launch
      compute_force_on_device<<<numBlocks, numThreadsInBlock>>>(d_forces, d_particles, N, g, d_force_qk);
      err = cudaGetLastError();
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to launch compute_force_on_device kernel (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}



      //Update positions and velocities according to force
      //Kernel launch
      compute_pos_and_vel_on_device<<<numBlocks, numThreadsInBlock>>>(d_forces, d_particles, delta_t, N, g);
      err = cudaGetLastError();
      if (err != cudaSuccess)
	{
	  fprintf(stderr, "Failed to launch compute_pos_and_vel_on_device kernel (error code %s)!\n", cudaGetErrorString(err));
	  exit(EXIT_FAILURE);
	}
    }
    /* End of computing */

    /* Time passed */
    end = getTime();
    total_time = end - start;

    //To make sure kernels have finished
    cudaThreadSynchronize();

    /* Copy the final states to Host */
    err = cudaMemcpy(particles, d_particles, size_particle, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
      {
	fprintf(stderr, "Failed to copy vector particles from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
      }

    /* print final states of particles */
    cout << "Final positions and velocities " << endl;
    print_particles(particles);


    int num_flops = 20 ;
    double flop_rate = n_steps * (1E-9 * N * N * num_flops) / total_time;
    double BW = n_steps * (N * N * sizeof(double) * 3.0)/total_time /N /N /N;

    /* output results */
    fprintf(stdout, " Time in seconds: %f \n", total_time);
    fprintf(stdout, " Gflops Rate: %f GFlop/s\n", flop_rate);
    fprintf(stdout, " Bandwidth Rate : %f GB/s\n", BW);


    /* cleanup */
    err = cudaFree(d_particles);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to free device vector d_particles (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    err = cudaFree(d_forces);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to free device vector d_forces (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    err = cudaFree(d_force_qk);
    if (err != cudaSuccess)
      {
        fprintf(stderr, "Failed to free device vector d_forces (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    free(particles);
    free(forces);
    free(force_qk);


    return 0;
}


__global__ void compute_pos_and_vel_on_device(vect_t *forces, particle_t *particles, int delta_t, int N, double g){
    int q = 0;
    q = blockIdx.x*blockDim.x + threadIdx.x;

    if(q>=0 && q<N){
      //position update
      particles[q].position[X] += delta_t * particles[q].velocity[X];
      particles[q].position[Y] += delta_t * particles[q].velocity[Y];

      //velocity update
      particles[q].velocity[X] += delta_t/particles[q].mass*forces[q][X];
      particles[q].velocity[Y] += delta_t/particles[q].mass*forces[q][Y];
    }
}


__global__ void compute_force_on_device(vect_t *forces, particle_t *particles, int N, double g, double *force_qk){
//It is the basic computation code, computing directly the forces.
    int q = 0;
    q = blockIdx.x*blockDim.x + threadIdx.x;

    if(q>=0 && q<N){
      for(int k = 0; k<N; k++){
	if(k!=q){
	  double x_diff = particles[q].position[X] - particles[k].position[X];
	  double y_diff = particles[q].position[Y] - particles[k].position[Y];
	  double dist = sqrt(x_diff*x_diff + y_diff*y_diff);
	  double dist_cubed = dist*dist*dist;
	  forces[q][X] -= g*particles[q].mass*particles[k].mass/dist_cubed * x_diff;
	  forces[q][Y] -= g*particles[q].mass*particles[k].mass/dist_cubed * y_diff;
	}
      }
    }
}



void print_particles(particle_t *particles)
{
    for (int part = 0; part < N; part++) {
        particle_t particle = particles[part];
        cout << "position of particle " << part << " is (" << particle.position[X] << "," << particle.position[Y]  << ")" << endl;
        cout << "velocity of particle " << part << " is (" << particle.velocity[X] << "," << particle.velocity[Y]  << ")" << endl;
    }
    cout << "------------------" << endl;
}


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void init_force(vect_t *vect, int N)
{
    int q=0;

    for(q=0; q < N; q++)
    {
        vect[q][X]=0.0;
        vect[q][Y]=0.0;
    }
}


void init_particle(particle_t *particles, int N)
{
    for (int i = 0; i < N; i++) {
        if(i==5){ //Sun
            struct particle_t particle;
            particle.mass = fRand(1.9890e+30, 1.9890e+30);
            particle.position[X] = fRand(0,0);
            particle.position[Y] = fRand(0,0);
            particle.velocity[X] = fRand(0,0);
            particle.velocity[Y] = fRand(0,0);
            particles[i] = particle;
        }else{
            struct particle_t particle;
            double min = 5.9740e+24;
            double max = 10*min;
            particle.mass = fRand(min, max);
            particle.position[X] = fRand(-2.50e+11, 2.50e+11);
            particle.position[Y] = fRand(-2.50e+11, 2.50e+11);
            particle.velocity[X] = fRand(0, 0);
            particle.velocity[Y] = fRand(2.9800e+03, 2.9800e+05);
            particles[i] = particle;
        }
    }
}


double getTime()
{
    const double kMicro = 1.0e-6;
    struct timeval TV;

    const int RC = gettimeofday(&TV, NULL);
    if(RC == -1)
    {
        printf("ERROR: Bad call to gettimeofday\n");
        return(-1);
    }
    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}


void read_input(particle_t *particles, int N, char *file)
{
  ifstream myfile;
  double mass, positionX, positionY, velocityX, velocityY;
  int i = 0;

  myfile.open(file, ios::in);

  if(myfile.is_open()){
    while(i < N){
      myfile >> positionX >> positionY >> velocityX >> velocityY >> mass;
      struct particle_t particle;
      particle.mass = mass;
      particle.position[X] = positionX;
      particle.position[Y] = positionY;
      particle.velocity[X] = velocityX;
      particle.velocity[Y] = velocityY;
      particles[i] = particle;
      i++;
    }
    myfile.close();
  }else{
    cout << fprintf(stderr, "error: file could not be opened for reading") << endl;
    exit(1);
  }
}

void usage(int argc, char **argv)
{
    cout << "Usage " << argv[0] << "<N> <delta_t> <T> <debug> <num_threads> <part_file>" << endl;
    cout << "\tN\t - number of particles (positive integer)" << endl;
    cout << "\tdelta_t\t - time difference between two time (positive)" << endl;
    cout << "\tT\t - final time number (positive)" << endl;
    cout << "\tdebug\t - 1 to see the states of particles time to time, 0 to see only final step" << endl;
    cout << "\tnum_threads\t - number of threads" << endl;
    cout << "\tpart_file\t - name of the file containing information the inital states of particles" << endl;
    exit(1);
}
