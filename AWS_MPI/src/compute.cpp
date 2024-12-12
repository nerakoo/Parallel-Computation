#include "stimulus.h"
#include "obstacle.h"
#include "compute.h"
#include "plotter.h"
#include <list>
#include <random>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

#ifdef _MPI_
#include <mpi.h>

#endif
#include <math.h>

///
/// Compute object
///
/// upon construction, runs a simulation.
///
Compute::Compute(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
		 const int seedv):
    u(u), plt(plt), cb(cb), myRank(_myRank), seedv(seedv), M(u.M), N(u.N), RANDSTIM(0)
{
#ifdef _MPI_
    // 补充的第一部分代码：
    // 创建 MPI 笛卡尔拓扑
    int dims[2] = {cb.py, cb.px}; // 进程拓扑的维度，网格尺寸
    int periods[2] = {0, 0};      // 非周期性边界，1表示周期性，0表示非周期性
    int coords[2];                // 进程在拓扑中的坐标
    MPI_Comm cartComm;            // 笛卡尔通信器

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartComm);
    MPI_Cart_coords(cartComm, myRank, 2, coords);

    // 获取相邻进程的 rank
    MPI_Cart_shift(cartComm, 0, 1, &upRank, &downRank);    // 纵向（行）方向
    MPI_Cart_shift(cartComm, 1, 1, &leftRank, &rightRank); // 横向（列）方向

    // 确定当前进程是否在全局边界上
    topGlobalEdge = (upRank == MPI_PROC_NULL);
    botGlobalEdge = (downRank == MPI_PROC_NULL);
    leftGlobalEdge = (leftRank == MPI_PROC_NULL);
    rightGlobalEdge = (rightRank == MPI_PROC_NULL);

    // 创建列类型，用于发送和接收列数据
    MPI_Type_vector(u.gridM, 1, u.gridN, MPI_DOUBLE, &columnType);
    MPI_Type_commit(&columnType);

#else    
    // 非 MPI 环境，所有边界都视为全局边界
    topGlobalEdge = true;
    botGlobalEdge = true;
    leftGlobalEdge = true;
    rightGlobalEdge = true;
#endif
}

void Compute::exchangeGhostCells()
{
#ifdef _MPI_
    int reqCount = 0; // 请求计数器

    // 发送和接收当前时间步的边界数据（u.curr）

    // 向上发送，向下接收
    if (upRank != MPI_PROC_NULL) {
        // 发送当前时间步的第一行给上面的进程
        MPI_Isend(u.cur(1, 0), u.gridN, MPI_DOUBLE, upRank, 0, MPI_COMM_WORLD, &sndRqst[reqCount]);
        // 接收来自上面进程的鬼魂单元行（存储在第 0 行）
        MPI_Irecv(u.cur(0, 0), u.gridN, MPI_DOUBLE, upRank, 1, MPI_COMM_WORLD, &rcvRqst[reqCount]);
        reqCount++;
    }
    if (downRank != MPI_PROC_NULL) {
        // 发送当前时间步的最后一行给下面的进程
        MPI_Isend(u.cur(u.gridM - 2, 0), u.gridN, MPI_DOUBLE, downRank, 1, MPI_COMM_WORLD, &sndRqst[reqCount]);
        // 接收来自下面进程的鬼魂单元行（存储在最后一行）
        MPI_Irecv(u.cur(u.gridM - 1, 0), u.gridN, MPI_DOUBLE, downRank, 0, MPI_COMM_WORLD, &rcvRqst[reqCount]);
        reqCount++;
    }

    // 向左发送，向右接收
    if (leftRank != MPI_PROC_NULL) {
        // 发送当前时间步的第一列给左侧的进程
        MPI_Isend(u.cur(0, 1), 1, columnType, leftRank, 2, MPI_COMM_WORLD, &sndRqst[reqCount]);
        // 接收来自左侧进程的鬼魂单元列（存储在第 0 列）
        MPI_Irecv(u.cur(0, 0), 1, columnType, leftRank, 3, MPI_COMM_WORLD, &rcvRqst[reqCount]);
        reqCount++;
    }
    if (rightRank != MPI_PROC_NULL) {
        // 发送当前时间步的最后一列给右侧的进程
        MPI_Isend(u.cur(0, u.gridN - 2), 1, columnType, rightRank, 3, MPI_COMM_WORLD, &sndRqst[reqCount]);
        // 接收来自右侧进程的鬼魂单元列（存储在最后一列）
        MPI_Irecv(u.cur(0, u.gridN - 1), 1, columnType, rightRank, 2, MPI_COMM_WORLD, &rcvRqst[reqCount]);
        reqCount++;
    }

    // 等待所有发送和接收完成
    MPI_Waitall(reqCount, sndRqst, MPI_STATUSES_IGNORE);
    MPI_Waitall(reqCount, rcvRqst, MPI_STATUSES_IGNORE);
#endif
}

    
///
/// Simulate
///
/// calls class specific calcU and calcEdgeU
///
void Compute::Simulate()
{
    const unsigned int t = 1;	 // timestep
    const unsigned int h = 1;    // grid space
    const double c = 0.29;   // velocity
    const double kappa = c*t/h;

    mt19937 generator(seedv);

    uniform_int_distribution<int> randRow(1, cb.m-10);
    uniform_int_distribution<int> randCol(1, cb.n-10);
    uniform_int_distribution<int> randEvent(0, 100);


    u.setAlpha((c*t/h) * (c*t/h));

    list<Stimulus *> sList;
    list<Obstacle *> oList;
    int iter = 0;

    ifstream f(cb.configFileName);
    if (cb.config.count("objects")){
	for (int i=0; i<cb.config["objects"].size(); i++){
	    auto const &ob = cb.config["objects"][i];
	    if (ob["type"] == "sine"){
		sList.push_back(new StimSine(u, ob["row"], ob["col"],
					     ob["start"], ob["duration"],
					     ob["period"]));
	    }else if (ob["type"] == "rectobstacle"){
		oList.push_back(new Rectangle(u, ob["row"], ob["col"],
					      ob["height"], ob["width"]));
	    }
	}
    }else{
	fprintf(stderr, "Using hardcoded stimulus\n");
	Rectangle obstacleA(u, cb.m/2+5, cb.n/2, 45, 5);
	Rectangle obstacleB(u, cb.m/2-50, cb.n/2, 45, 5);
	sList.push_back(new StimSine(u, cb.m/2, cb.n/3, 0 /*start*/, 500/*duration*/, 10 /*period*/));
    }

    ///
    /// generate stimulus
    ///
    /// once quiet (non-deterministic),
    /// we exit this loop and go into a loop that
    /// continues until iterations is exhausted
    ///
    while (!sList.empty() && iter < cb.niters){
	for (auto it = begin(sList); it!= end(sList);){
	    if (!(*it)->doit(iter)){
		delete *it;
		it = sList.erase(it);
	    }else{
		it++;
	    }
	}

    // u.printMap2(iter);

    #ifdef _MPI_
        exchangeGhostCells(); // 交换鬼魂单元数据
    #endif

	calcU(u);

    #ifdef _MPI_
        // 在计算边界之前，确保鬼魂单元数据已更新
        // 可以再次交换必要的数据（如果需要）
    #endif

	calcEdgeU(u, kappa);

	if (cb.plot_freq && iter % cb.plot_freq == 0)
	    plt->updatePlot(iter, u.gridM, u.gridN);

	// DEBUG start
	//	u.printActive(iter);
	// DEBUG end

	u.AdvBuffers();
	iter++;
    }

    ///
    /// all stimulus done
    /// keep simulating till end
    ///
    for (;iter < cb.niters; iter++){
    #ifdef _MPI_
        exchangeGhostCells(); // 交换鬼魂单元数据
    #endif
	calcU(u);
	//	if (cb.plot_freq && iter % cb.plot_freq == 0)
	//	    plt->updatePlot(iter, u.gridM, u.gridN);

    #ifdef _MPI_
        // 在计算边界之前，确保鬼魂单元数据已更新
        // 可以再次交换必要的数据（如果需要）
    #endif

	calcEdgeU(u, kappa);
	if ((cb.plot_freq!=0) && (iter % cb.plot_freq == 0))
	    plt->updatePlot(iter, u.gridM, u.gridN);


	// DEBUG
	// u.printActive(iter);
	u.AdvBuffers();
    }
}

TwoDWave::TwoDWave(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
		 const int seedv):
    Compute(u, plt, cb, _myRank, seedv){};

///
/// compute the interior cells
///
///
void TwoDWave::calcU(Buffers &u)
{

    // interior always starts at 2,2, ends at gridN
    for (int i=1; i<u.gridM-1; i++){
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(i,j) =
		u.alpV(i,j) *
		(u.curV(i-1,j) + u.curV(i+1,j) +
		 u.curV(i,j-1) + u.curV(i,j+1) - 4 * u.curV(i,j)) +
		2 * u.curV(i,j) - u.preV(i,j);
	}
    }
}

// /
// / compute edges
// /
// / compute interior edges. These are not ghost cells but cells that rely
// / on either ghost cell values or boundary cell values.
// /
void TwoDWave::calcEdgeU(Buffers &u, const double kappa)
{

    // top and bottom edge
    for (int j=1; j<u.gridN-1; j++){
        int i = 1;
        *u.nxt(i,j) =
            u.alpV(i,j) *
            (u.curV(i-1,j) + u.curV(i+1,j) +
             u.curV(i,j-1) + u.curV(i,j+1) - 4* u.curV(i,j)) +
            2 * u.curV(i,j) - u.preV(i,j);
        i = u.gridM-2;
        *u.nxt(i,j) =
                u.alpV(i,j) *
                (u.curV(i-1,j) + u.curV(i+1,j) +
                 u.curV(i,j-1) + u.curV(i,j+1) - 4* u.curV(i,j)) +
                2 * u.curV(i,j) - u.preV(i,j);
    }

    // left and right
    for (int i=1; i<u.gridM-1; i++){
        int j = 1;
        *u.nxt(i,j) =
            u.alpV(i,j) *
            (u.curV(i-1,j) + u.curV(i+1,j) +
             u.curV(i,j-1) + u.curV(i,j+1) - 4*u.curV(i,j)) +
            2 * u.curV(i,j) - u.preV(i,j);
        j = u.gridN-2;
        *u.nxt(i,j) =
            u.alpV(i,j) *
            (u.curV(i-1,j) + u.curV(i+1,j) +
             u.curV(i,j-1) + u.curV(i,j+1) - 4*u.curV(i,j)) +
            2 * u.curV(i,j) - u.preV(i,j);
    }

    // set the boundary conditions to absorbing boundary conditions (ABC)
    // du/dx = -1/c du/dt   x=0
    // du/dx = 1/c du/dt    x=N-1
    // conditions for an internal boundary (ie.g. ghost cells)
    // top edge

    // top global edge (instead of ghost cells)
    if (topGlobalEdge){
	// top row absorbing boundary condition
	int i = 0;
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(i,j) = u.curV(i+1,j) +
		((kappa-1)/(kappa+1)) * (u.nxtV(i+1,j) - u.curV(i,j));
	}
    }

    // bottom edge (instead of ghost cells)
    if (botGlobalEdge){
	int i = u.gridM-1;
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(i,j) = u.curV(i-1,j) +
		((kappa-1)/(kappa+1)) * (u.nxtV(i-1,j) - u.curV(i,j));
	}
    }

    // left edge
    if (leftGlobalEdge){
	int j = 0;
	for (int i=1; i<u.gridM-1; i++){
	    *u.nxt(i,j) = u.curV(i,j+1) +
		((kappa-1)/(kappa+1)) * (u.nxtV(i,j+1) - u.curV(i,j));
	}
    }
    // right edge
    if (rightGlobalEdge){
	int j = u.gridN-1;
	for (int i=1; i<u.gridM-1; i++){
	    *u.nxt(i,j) = u.curV(i,j-1) +
		((kappa-1)/(kappa+1)) * (u.nxtV(i,j-1) - u.curV(i,j));
	}
    }
}

// void TwoDWave::calcEdgeU(Buffers &u, const double kappa)
// {
//     // 计算内部边界网格点（不在全局边界上）
//     // 顶部和底部边界
//     for (int j=1; j<u.gridN-1; j++){
//         int i;
//         // 顶部边界
//         i = 1;
//         if (!topGlobalEdge) {
//             *u.nxt(i,j) =
//                 u.alpV(i,j) *
//                 (u.curV(i-1,j) + u.curV(i+1,j) +
//                  u.curV(i,j-1) + u.curV(i,j+1) - 4 * u.curV(i,j)) +
//                 2 * u.curV(i,j) - u.preV(i,j);
//         }
//         // 底部边界
//         i = u.gridM - 2;
//         if (!botGlobalEdge) {
//             *u.nxt(i,j) =
//                 u.alpV(i,j) *
//                 (u.curV(i-1,j) + u.curV(i+1,j) +
//                  u.curV(i,j-1) + u.curV(i,j+1) - 4 * u.curV(i,j)) +
//                 2 * u.curV(i,j) - u.preV(i,j);
//         }
//     }

//     // 左右边界
//     for (int i=1; i<u.gridM-1; i++){
//         int j;
//         // 左边界
//         j = 1;
//         if (!leftGlobalEdge) {
//             *u.nxt(i,j) =
//                 u.alpV(i,j) *
//                 (u.curV(i-1,j) + u.curV(i+1,j) +
//                  u.curV(i,j-1) + u.curV(i,j+1) - 4 * u.curV(i,j)) +
//                 2 * u.curV(i,j) - u.preV(i,j);
//         }
//         // 右边界
//         j = u.gridN - 2;
//         if (!rightGlobalEdge) {
//             *u.nxt(i,j) =
//                 u.alpV(i,j) *
//                 (u.curV(i-1,j) + u.curV(i+1,j) +
//                  u.curV(i,j-1) + u.curV(i,j+1) - 4 * u.curV(i,j)) +
//                 2 * u.curV(i,j) - u.preV(i,j);
//         }
//     }

//     // 对于全局边界，应用吸收边界条件（ABC）
//     const double coeff = (kappa - 1) / (kappa + 1);

//     // 顶部全局边界
//     if (topGlobalEdge) {
//         int i = 1;
//         for (int j = 1; j < u.gridN - 1; j++) {
//             *u.nxt(i-1, j) = u.curV(i, j) +
//                 coeff * (*u.nxt(i, j) - u.curV(i-1, j));
//         }
//     }

//     // 底部全局边界
//     if (botGlobalEdge) {
//         int i = u.gridM - 2;
//         for (int j = 1; j < u.gridN - 1; j++) {
//             *u.nxt(i+1, j) = u.curV(i, j) +
//                 coeff * (*u.nxt(i, j) - u.curV(i+1, j));
//         }
//     }

//     // 左侧全局边界
//     if (leftGlobalEdge) {
//         int j = 1;
//         for (int i = 1; i < u.gridM - 1; i++) {
//             *u.nxt(i, j-1) = u.curV(i, j) +
//                 coeff * (*u.nxt(i, j) - u.curV(i, j-1));
//         }
//     }

//     // 右侧全局边界
//     if (rightGlobalEdge) {
//         int j = u.gridN - 2;
//         for (int i = 1; i < u.gridM - 1; i++) {
//             *u.nxt(i, j+1) = u.curV(i, j) +
//                 coeff * (*u.nxt(i, j) - u.curV(i, j+1));
//         }
//     }
// }

//!
//! Use a different propgation model
//! This model shifts values in the horizontal direction
//!
DebugPropagate::DebugPropagate(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
		 const int seedv):
    Compute(u, plt, cb, _myRank, seedv){};

//!
//! compute the interior cells
//!
void DebugPropagate::calcU(Buffers &u)
{

    // interior always starts at 2,2, ends at gridN-3
    for (int i=2; i<u.gridM-2; i++){
	for (int j=2; j<u.gridN-2; j++){
	    *u.nxt(i,j) = u.curV(i, j-1);
	}
    }
}

//!
//! compute edges
//! (either interior edges or global edges)
//!
void DebugPropagate::calcEdgeU(Buffers &u, const double kappa)
{
    if (topGlobalEdge){
	// top row absorbing boundary condition
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(1,j) = 0;
	}
    }else{
	int i = 1;
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(i,j) = u.curV(i, j-1);
	}
    }

    // bottom edge
    if (botGlobalEdge){
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(u.gridM-2,j) = 0;
	}
    }else{
	int i=u.gridM-2;
	for (int j=1; j<u.gridN-1; j++){
	    *u.nxt(i,j) = u.curV(i, j-1);
	}
    }

    // left edge
    if (leftGlobalEdge){
	for (int i=1; i<u.gridM-1; i++){
	    *u.nxt(i,1) = 0.0;
	}
    }else{
	int j=1;
	for (int i=1; i<u.gridM-1; i++){
	    *u.nxt(i,j) = u.curV(i, j-1);
	}
    }
    // right edge
    if (rightGlobalEdge){
	for (int i=1; i<u.gridM-1; i++){
	    // right column
	    *u.nxt(i,u.gridN-2) = 0.0;
	}
    }else{
	int j=u.gridN-2;
	for (int i=1; i<u.gridM-1; i++){
	    *u.nxt(i,j) = u.curV(i, j-1);
	}
    }
}


