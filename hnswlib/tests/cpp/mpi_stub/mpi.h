#pragma once

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;

static const MPI_Datatype MPI_BYTE = 1;
static const MPI_Datatype MPI_DOUBLE = 2;
static const MPI_Datatype MPI_UNSIGNED_LONG_LONG = 3;
static const MPI_Datatype MPI_FLOAT = 4;
static const MPI_Datatype MPI_UINT32_T = 5;
static const MPI_Op MPI_MIN = 6;
static const MPI_Op MPI_MAX = 7;
static const MPI_Op MPI_SUM = 8;
static const MPI_Comm MPI_COMM_WORLD = 9;
static const int MPI_THREAD_FUNNELED = 10;

double MPI_Wtime(void);
int MPI_Reduce(
    const void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
int MPI_Gather(
    const void*, int, MPI_Datatype,
    void*, int, MPI_Datatype, int, MPI_Comm);
int MPI_Barrier(MPI_Comm);
int MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm);
int MPI_Comm_rank(MPI_Comm, int*);
int MPI_Comm_size(MPI_Comm, int*);
int MPI_Init_thread(int*, char***, int, int*);
int MPI_Finalize(void);
