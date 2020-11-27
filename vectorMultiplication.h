// Copyright 2020 Alekhin Denis
#ifndef VECTOR_MULTIPLICATION_H_
#define VECTOR_MULTIPLICATION_H_

#include <vector>
#include <queue>

void getRandomVector(std::vector<double>* vector, 
  int size = 10000, int min = 0, int max = 1000);

void printVector(const std::vector<double>* vector);

void vectorMultiplicationSequential(std::vector<double>* vector, double x);

void vectorMultiplicationParallelGroup(std::vector<double>* vector, double x);




















struct Task {
  std::vector<double> vector;
  double x;
  int numberOfTask;

  Task(std::vector<double> _vector, double _x, int _numberOfTask) {
    vector = _vector;
    x = _x;
    numberOfTask = _numberOfTask;
  }
};

class TaskParallelism {
private:
  std::queue<Task> task;
  int segmentationSize; // кол-во элементов в каждом задании
  int countOfTasks;
  std::vector<bool> busyProc; // занят ли процесс
  std::vector<double> result;

public:
  std::vector<double> getVector() { return result; }

  TaskParallelism::TaskParallelism(const std::vector<double>* vector,
    double x, int _segmentationSize);

  std::vector<double> TaskParallelism::getSubvector(
    const std::vector<double>* vector, int numberOfTask);

  void sendTask(Task task, int proc);
  void solveTask();
  void recvResult(); // 
  void scheduler(); // распределение задач по процессам
};

void vectorMultiplicationParallelQueue(std::vector<double>* vector, double x);

#endif  // VECTOR_MULTIPLICATION_H_
