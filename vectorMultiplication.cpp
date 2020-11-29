// Copyright 2020 Alekhin Denis
#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <ctime>
#include "../../modules/task_2/VectorMPI/vectorMultiplication.h"

void getRandomVector(std::vector<double>* vector, int size, int min, int max) {
  if (size < 1 || min < 0 || max < min)
    throw "Error";
  
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < size; i++) {
    vector->push_back(gen() % (max - min + 1) + min);
  }
}

void printVector(const std::vector<double>* vector) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (int i = 0; i < vector->size(); i++) {
      std::cout << (*vector)[i] << std::endl;
    }
  }
}

void vectorMultiplicationSequential(std::vector<double>* vector, double x) {
  for (int i = 0; i < vector->size(); i++) {
    (*vector)[i] *= x;
  }
}

void vectorMultiplicationParallelGroup(std::vector<double>* vector, double x) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&x,
    1,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);

  int* sendcounts = new int[size];
  int* displs = new int[size];
  int sum = 0;
  int max = -1;
  for (int i = 0; i < size; i++) {
    if (i != size - 1) {
      sendcounts[i] = ceil(vector->size() / size);
    } else {
      sendcounts[i] = vector->size() - ceil(vector->size() / size) * (size - 1);
    }

    displs[i] = sum;
    sum += sendcounts[i];
    if (sendcounts[i] > max) {
      max = sendcounts[i];
    }
  }

  std::vector<double> local_vector(sendcounts[rank]);

  MPI_Scatterv(vector->data(),
    sendcounts,
    displs,
    MPI_DOUBLE,
    local_vector.data(),
    sendcounts[rank],
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);
  
  for (int i = 0; i < local_vector.size(); i++) {
    local_vector[i] *= x;
  }
  
  MPI_Gatherv(local_vector.data(),
    sendcounts[rank],
    MPI_DOUBLE,
    vector->data(),
    sendcounts,
    displs,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);
    
}
















void vectorMultiplicationParallelQueue(std::vector<double>* vector, double x, int _segmentationSize) {
  TaskParallelism taskParallelism(vector, x, _segmentationSize);
  taskParallelism.scheduler();
  *vector = taskParallelism.getVector();
}

std::vector<double> TaskParallelism::getSubvector(
  const std::vector<double>* vector, int numberOfTask) {
  std::vector<double> result(segmentationSize);
  std::copy(
    vector->begin() + numberOfTask * segmentationSize,
    vector->begin() + numberOfTask * segmentationSize + segmentationSize,
    result.begin());

  return result;
}

TaskParallelism::TaskParallelism(const std::vector<double>* vector,
  double x, int _segmentationSize) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  segmentationSize = _segmentationSize;
  countOfTasks = vector->size() / segmentationSize;
  
  for (int i = 0; i < vector->size() / segmentationSize; i++) {
    task.push(Task(getSubvector(vector, i), x, i));
  }
  
  busyProc.resize(size, false);
  busyProc[0] = true;
}

void TaskParallelism::scheduler() {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // initialization processes with tasks
  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      if (!task.empty() && !busyProc[i]) {
        busyProc[i] = true;
        sendTask(task.front(), i);
        task.pop();
      }
    }
  }

  // send more task to processes 
  if(rank == 0) {
    while (result.size() != segmentationSize * countOfTasks) {
      sendTask(task.front(), recvResult());
    }
  } else {
    while (solveTask() != 1);
  }

  // send processes signal, that there is no more tasks
  if (rank == 0) {
    for (int i = 1; i < size; i++) {
      noMoreTaskSignal(i);
    }
  }
}

void TaskParallelism::sendTask(Task task, int proc) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int noMoreTask = 0;

  MPI_Send(
    &noMoreTask,
    1,
    MPI_INT,
    proc,
    1,
    MPI_COMM_WORLD);

  task.vector.push_back(task.x);
  MPI_Send(
    &task.vector[0],
    task.vector.size(),
    MPI_DOUBLE,
    proc,
    0,
    MPI_COMM_WORLD);
}

void TaskParallelism::noMoreTaskSignal(int proc) {
  int noMoreTask = 1;

  MPI_Send(
    &noMoreTask,
    1,
    MPI_INT,
    proc,
    1,
    MPI_COMM_WORLD);
}

int TaskParallelism::solveTask() {
  std::vector<double> local_vector(segmentationSize + 1);

  int noMoreTask = 0;
  MPI_Recv(
    &noMoreTask,
    1,
    MPI_INT,
    0,
    1,
    MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);

  if (noMoreTask == 1) {
    return 1;
  }

  MPI_Request request;
  MPI_Recv(
    &local_vector[0],
    local_vector.size(),
    MPI_DOUBLE,
    0,
    0,
    MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);

  for (int i = 0; i < local_vector.size() - 1; i++) {
    local_vector[i] *= local_vector[local_vector.size() - 1];
  }
  local_vector.pop_back();

  MPI_Send(
    &local_vector[0],
    local_vector.size(),
    MPI_DOUBLE,
    0,
    0,
    MPI_COMM_WORLD);

  return 0;
}

int TaskParallelism::recvResult() {
  std::vector<double> recv(segmentationSize);
  MPI_Status status;

  MPI_Recv(&recv[0],
    segmentationSize,
    MPI_DOUBLE,
    MPI_ANY_SOURCE,
    0,
    MPI_COMM_WORLD,
    &status);

  result.insert(result.end(), recv.begin(), recv.end());
  busyProc[status.MPI_SOURCE] = false;
  return status.MPI_SOURCE;
}
