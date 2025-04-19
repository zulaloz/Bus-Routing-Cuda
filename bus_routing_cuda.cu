// 🚍 Otobüs Rotalama Optimizasyonu - C++ ve CUDA Hazırlık
// Amaç: Popülasyon üzerinden genetik algoritma ve paralel fitness hesaplama

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define NUM_STOPS 10
#define POP_SIZE 30
#define MAX_GENERATIONS 100

struct Individual {
    int bus_id;
    int route[NUM_STOPS];
    int departure_time;
};

int distance_matrix[NUM_STOPS][NUM_STOPS];
int passenger_demand[NUM_STOPS];
int bus_capacities[5] = {40, 30, 50, 45, 35};

float fitness_scores[POP_SIZE];

void generate_distance_matrix() {
    for (int i = 0; i < NUM_STOPS; ++i) {
        for (int j = 0; j < NUM_STOPS; ++j) {
            distance_matrix[i][j] = rand() % 10 + 1;
        }
    }
}

void generate_passenger_demand() {
    for (int i = 0; i < NUM_STOPS; ++i) {
        passenger_demand[i] = rand() % 11 + 5;
    }
}

Individual create_individual() {
    Individual ind;
    ind.bus_id = rand() % 5;
    for (int i = 0; i < NUM_STOPS; ++i) ind.route[i] = i;
    std::random_shuffle(ind.route, ind.route + NUM_STOPS);
    ind.departure_time = rand() % 5 + 6;
    return ind;
}

void create_population(std::vector<Individual>& population) {
    for (int i = 0; i < POP_SIZE; ++i) {
        population.push_back(create_individual());
    }
}

__device__ float calculate_fitness_gpu(const Individual& ind, const int* dist_matrix, const int* demand, const int* capacities) {
    int total_distance = 0;
    int total_passengers = 0;
    for (int i = 0; i < NUM_STOPS - 1; ++i) {
        int from = ind.route[i];
        int to = ind.route[i + 1];
        total_distance += dist_matrix[from * NUM_STOPS + to];
    }
    for (int i = 0; i < NUM_STOPS; ++i) {
        total_passengers += demand[ind.route[i]];
    }
    int capacity = capacities[ind.bus_id];
    int penalty = (total_passengers > capacity) ? (total_passengers - capacity) * 10 : 0;
    return 1.0f / (total_distance + 1 + penalty);
}

__global__ void calculate_fitness_kernel(Individual* population, float* fitness, const int* dist_matrix, const int* demand, const int* capacities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POP_SIZE) {
        fitness[idx] = calculate_fitness_gpu(population[idx], dist_matrix, demand, capacities);
    }
}

int main() {
    srand(time(0));
    std::vector<Individual> population;
    generate_distance_matrix();
    generate_passenger_demand();
    create_population(population);

    // CUDA için bellek tahsisi ve veri kopyalama
    Individual* d_population;
    float* d_fitness;
    int* d_dist_matrix;
    int* d_demand;
    int* d_capacities;

    cudaMalloc(&d_population, sizeof(Individual) * POP_SIZE);
    cudaMalloc(&d_fitness, sizeof(float) * POP_SIZE);
    cudaMalloc(&d_dist_matrix, sizeof(int) * NUM_STOPS * NUM_STOPS);
    cudaMalloc(&d_demand, sizeof(int) * NUM_STOPS);
    cudaMalloc(&d_capacities, sizeof(int) * 5);

    cudaMemcpy(d_population, population.data(), sizeof(Individual) * POP_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_matrix, distance_matrix, sizeof(int) * NUM_STOPS * NUM_STOPS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_demand, passenger_demand, sizeof(int) * NUM_STOPS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacities, bus_capacities, sizeof(int) * 5, cudaMemcpyHostToDevice);

    // CUDA kernel çağır
    calculate_fitness_kernel<<<(POP_SIZE + 255)/256, 256>>>(d_population, d_fitness, d_dist_matrix, d_demand, d_capacities);
    cudaDeviceSynchronize();

    // Sonuçları host'a kopyala
    cudaMemcpy(fitness_scores, d_fitness, sizeof(float) * POP_SIZE, cudaMemcpyDeviceToHost);

    std::cout << "Ilk bireyin fitness puani (GPU): " << fitness_scores[0] << std::endl;

    // Bellek temizle
    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_dist_matrix);
    cudaFree(d_demand);
    cudaFree(d_capacities);

    return 0;
}