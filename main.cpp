// Copyright 2020 Alekhin Denis
#include <stdlib.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "../../modules/task_2/VectorMPI/vectorMultiplication.h"

TEST(Get_Random_Vector, Get_Random_Vector_Test) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    EXPECT_NO_THROW(getRandomVector(&test, 20, 3, 5));
  }
}

TEST(Get_Random_Vector, Get_Random_Vector_Throw_Error_Test1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    EXPECT_ANY_THROW(getRandomVector(&test, 10, 6, 5));
  }
}

TEST(Get_Random_Vector, Get_Random_Vector_Throw_Error_Test2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    EXPECT_ANY_THROW(getRandomVector(&test, 0, 3, 5));
  }
}

TEST(Get_Random_Vector, Get_Random_Vector_Throw_Error_Test3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    EXPECT_ANY_THROW(getRandomVector(&test, 20, -1, 5));
  }
}

TEST(Vector_Multiplication_Sequantion, Vector_Multiplication_Sequantion_Works) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    getRandomVector(&test, 1000000, 0, 9);

    EXPECT_NO_THROW(vectorMultiplicationSequential(&test, 3));
  }
}

TEST(Vector_Multiplication_Sequantion, Vector_Multiplication_Sequantion_Test) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<double> test;
    getRandomVector(&test, 1000000, 0, 9);

    double startTime = MPI_Wtime();
    vectorMultiplicationSequential(&test, 3);
    double endTime = MPI_Wtime();

    std::cout << endTime - startTime << std::endl;
  }
}

TEST(Vector_Multiplication_Parallel_Group, Vector_Multiplication_Parallel_Group_Works) {
  std::vector<double> test;
  getRandomVector(&test, 10, 0, 9);
  
  EXPECT_NO_THROW(vectorMultiplicationParallelGroup(&test, 3));
}

TEST(Vector_Multiplication_Parallel_Group, Vector_Multiplication_Parallel_Group_Test) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> test;
  getRandomVector(&test, 1000000, 0, 9);

  double startTime = MPI_Wtime();
  vectorMultiplicationParallelGroup(&test, 3);
  double endTime = MPI_Wtime();
  if (rank == 0) {
    std::cout << endTime - startTime << std::endl;
  }
}










TEST(Vector_Multiplication_Parallel_Queue, Vector_Multiplication_Parallel_Group_Works) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> test;
  getRandomVector(&test, 10, 0, 9);
  if (size > 1) {
    EXPECT_NO_THROW(vectorMultiplicationParallelQueue(&test, 3));
    printVector(&test);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);

  return RUN_ALL_TESTS();
}
