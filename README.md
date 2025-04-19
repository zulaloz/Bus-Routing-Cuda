""# 🚌 Bus Routing Optimization with Genetic Algorithm and CUDA

This project implements a parallelized genetic algorithm using CUDA to optimize bus routing and departure schedules. It was originally developed as a graduation project to improve the efficiency of public transportation systems.

---

## 📌 Project Overview

- Each **individual** in the population represents a bus route and its departure time.
- The **goal** is to minimize the total travel distance while ensuring bus capacity constraints are respected.
- The **fitness function** penalizes solutions where total passenger demand exceeds the bus capacity.
- The **fitness calculations** are offloaded to the GPU using CUDA for faster performance.

---

## ⚙️ Technologies

- **C++**
- **CUDA**
- Parallel programming
- Genetic algorithms

---

## 🧠 Key Components

- `Individual`: A data structure representing one bus, its route, and departure time.
- `calculate_fitness_kernel`: CUDA kernel that computes fitness scores in parallel.
- Capacity constraints are enforced via penalty in the fitness function.

---

## 🚀 How to Run

You need **NVIDIA CUDA Toolkit** installed to compile and run this project.

```bash
nvcc -o bus_routing bus_routing_cuda.cu
./bus_routing
