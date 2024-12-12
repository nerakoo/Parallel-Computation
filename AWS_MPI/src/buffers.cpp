#include "buffers.h"
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

// Buffers::Buffers(ControlBlock& _cb, int _myRank):
//     cb(_cb),
//     myRank(_myRank)
// {
//     if (cb.px * cb.py == 1){
// 	// uniprocessor
// 	M = cb.m;
// 	N = cb.n;
// 	// we add extra ghost cells even though we don't need
// 	// them as this will keep the coordinate system the same
// 	// in the rest of the code.
// 	gridM = cb.m + 2;   // extra ghost cells not needed
// 	gridN = cb.n + 2;   // extra ghost cells not needed 

//     } else {
//         // set M, N and gridM and gridN as approrpiate 
// 	// for stencil method on MPI
// 	//

// 	N = cb.n / cb.px + (getExtraCol() ?  1 : 0);
// 	M = cb.m / cb.py + (getExtraRow() ?  1 : 0);
// 	gridN = N+2;   // add layer of ghost cells on left and right
// 	gridM = M+2;   // add layer of ghost cells on top and bottom
// 		printf("DEBUG : rank %d M, N = %d, %d,\n", myRank, M, N);
// 	//	fflush(stdout);
//     }    

//     // calculate global row and column origin for each rank
//     for (int col = 0, startC = 0; col< cb.px; col++){
// 	startCols.push_back(startC);
// 	startC += cb.n / cb.px + (getExtraCol(col, cb.n, cb.px) ?  1 : 0);
// 	fflush(stdout);
//     }
//     for (int row = 0, startR = 0; row< cb.py; row++){
// 	startRows.push_back(startR);
// 	startR += cb.m / cb.py + (getExtraRow(row, cb.m, cb.py) ? 1 : 0);
// 	fflush(stdout);
//     }
// };


Buffers::Buffers(ControlBlock& _cb, int _myRank):
    cb(_cb),
    myRank(_myRank)
{
    int px = cb.px;  // x 方向的进程数
    int py = cb.py;  // y 方向的进程数
    int n = cb.n;    // 全局网格的列数
    int m = cb.m;    // 全局网格的行数

    if (px * py == 1){
        // 单进程
        M = m;
        N = n;
        gridM = m + 2;   // 包含鬼魂单元
        gridN = n + 2;   // 包含鬼魂单元
        startRow = 0;
        startCol = 0;
    } else {
        // 多进程，计算每个进程的网格尺寸和起始行列
        int myRow = myRank / px; // 当前进程在进程网格中的行坐标
        int myCol = myRank % px; // 当前进程在进程网格中的列坐标

        // 计算基本的本地网格尺寸（不包括鬼魂单元）
        int baseM = m / py;       // 每个进程的基本行数
        int extraRows = m % py;   // 需要多分配一行的进程数量
        M = baseM + (myRow < extraRows ? 1 : 0);

        int baseN = n / px;       // 每个进程的基本列数
        int extraCols = n % px;   // 需要多分配一列的进程数量
        N = baseN + (myCol < extraCols ? 1 : 0);

        gridM = M + 2;   // 包含鬼魂单元
        gridN = N + 2;   // 包含鬼魂单元

        // 计算当前进程负责的全局起始行列（不包括鬼魂单元）
        startRow = myRow * baseM + std::min(myRow, extraRows);
        startCol = myCol * baseN + std::min(myCol, extraCols);

        // 调试信息
        // printf("Rank %d (myRow=%d, myCol=%d): M=%d, N=%d, startRow=%d, startCol=%d\n",
        //        myRank, myRow, myCol, M, N, startRow, startCol);
    }
}

void Buffers::setAlpha(double aVal){
    for (int i=0; i<gridM; i++){
	for (int j=0; j<gridN; j++){
	    *alp(i,j) = aVal;
	}
    }
}

///
/// print for debug purposes
///
void Buffers::print(int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    for (int i=0; i<gridM; i++){
	for (int j=0; j<gridN; j++){
	    printf("%2.3f ", curV(i, j));
	}
	printf("\n");
    }
    printf("--------------------------------\n");
}

///
/// print for debug purposes
///
void Buffers::printAlpha(){
    printf("%d  --------------------------------\n", myRank);
    for (int i=0; i<gridM; i++){
	for (int j=0; j<gridN; j++){
	    printf("%2.3f ", alpV(i, j));
	}
	printf("\n");
    }
    printf("--------------------------------\n");
}



///
/// printMap for debug purposes
///
/// '.' for 0 cells
/// '-' if magnitude is >0 but less than <1.0
/// '*' if magnitude is >= 1.0
///
void Buffers::printMap(int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    for (int i=0; i<gridM; i++){
	for (int j=0; j<gridN; j++){
	    double v = curV(i,j);
	    char c;
	    v = (v < 0.0) ? -1.0*v : v;
	    if (v == 0.0) {
		c = '.';
	    }else if (v < 1.0){
		c = '-';
	    }else
		c = '*';
	    printf("%c", c);
	}
	printf("\n");
    }
    printf("--------------------------------\n");
}

void Buffers::printMap2(int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    for (int i=0; i<gridM; i++){
	for (int j=0; j<gridN; j++){
	    double v = curV(i,j);
	    printf("%f ", v);
	}
	printf("\n");
    }
    printf("--------------------------------\n");
}

///
/// printActive for debug
///
/// prints coordinates for each cell that is not 0.
///
void Buffers::printActive(int iter){

    for (int i=1; i<gridM-1; i++){
	for (int j=1; j<gridN-1; j++){
	    std::pair<int, int> glob = mapToGlobal(i, j);
	    if (nxtV(i,j) != 0.0){
		// print coordinates in native global #s
		// no ghost cells
		//		printf("%04d, %3d, %3d, %.12f %d\n",
		//     iter, glob.first, glob.second,
		//     nxtV(i, j), myRank);
		printf("%02d %04d, %3d, %3d, %.12f\n",
		       myRank, iter, glob.first, glob.second,
		       nxtV(i, j));
		fflush(stdout);
	    }
	}
    }
}

///
/// sumSq
///
/// calculate sum of the squares of each cell
/// between [r,c] and (rend, cend)
///
double Buffers::sumSq(int r, int c, int rend, int cend){
    double sumSq = 0.0;
    for (int i=r; i<rend; i++){
	for (int j=c; j<cend; j++){
	    double v=curV(i,j);
	    sumSq += v * v;
	}
    }
    return sumSq;
}

///
/// mapToLocal
///
/// map global coord that don't include ghost cells
/// to local coordinates that assume ghost cells.
/// returns -1, -1 if coordinates are not in this buffer
/// 在并行计算中，每个进程只负责全局网格的一部分，该函数用于将全局坐标转换为当前进程本地的坐标，便于访问本地数据。
std::pair<int, int> Buffers::mapToLocal(int globr, int globc){
    // 将全局坐标 (globr, globc) 映射为本地坐标 (localRow, localCol)
    // 本地坐标包括鬼魂单元，因此需要加 1
    int localRow = globr - startRow + 1; // 加 1 因为鬼魂单元
    int localCol = globc - startCol + 1; // 加 1 因为鬼魂单元

    // 检查坐标是否在本地网格范围内（不包括鬼魂单元）
    if (localRow >= 1 && localRow <= M && localCol >= 1 && localCol <= N) {
        return std::pair<int, int>(localRow, localCol); // 返回本地坐标
    } else {
        return std::pair<int, int>(-1, -1); // 不在本地网格范围内
    }
}

///
/// mapToGlobal
///
/// map local coord that assumes ghost cells to
/// global coord that has no ghost cells
///
/// 将本地网格坐标（包含鬼魂单元）映射到全局网格坐标（不包含鬼魂单元）。在需要进行全局数据通信或结果汇总时，使用该函数将本地坐标转换为全局坐标。
std::pair<int, int> Buffers::mapToGlobal(int r, int c){
    // 将本地坐标 (r, c) 映射为全局坐标 (globalRow, globalCol)
    // 本地坐标包括鬼魂单元，因此需要减 1
    int globalRow = r + startRow - 1;
    int globalCol = c + startCol - 1;
    return std::pair<int, int>(globalRow, globalCol); // 返回全局坐标
}


///
/// check to see if r and c are contained in this buffer
///
/// r and c are global coordinates
///
bool Buffers::chkBounds(int r, int c){
    bool inRowBounds = (r >= startRow) && (r < startRow + M);
    bool inColBounds = (c >= startCol) && (c < startCol + N);

    return inRowBounds && inColBounds;
}



///
/// ArrBuff - constructor
///
/// allocate the memory pool
///
ArrBuff::ArrBuff(ControlBlock& _cb, int _myRank) :
    Buffers(_cb, _myRank),
    memoryPool(nullptr),
    alpha(nullptr),
    next(nullptr),
    curr(nullptr),
    prev(nullptr)

{
    memoryPool = new double[3 * gridM * gridN]();
    alpha = new double[gridM * gridN]();;
    u0 = memoryPool;
    u1 = &memoryPool[gridM * gridN];
    u2 = &memoryPool[2 * gridM *gridN];
    prev = u0;
    curr = u1;
    next = u2;
}


ArrBuff::~ArrBuff() {
    delete[] memoryPool;
    delete[] alpha;
}


///
/// ArrBuff AdvBuffers - rotate the pointers
///
void ArrBuff::AdvBuffers(){
    double *t;
    t = prev;
    prev = curr;
    curr = next;
    next = t;
}


///
/// Array of Structures
/// 
/// This class groups all the data for each i, j point
/// near each other in memory.
///
AofSBuff::AofSBuff(ControlBlock& _cb, int _myRank) :
    Buffers(_cb, _myRank),
    memoryPool(nullptr),
    alpha(3),
    next(2),
    current(1),
    previous(0)
{
    memoryPool = new point[gridM * gridN]();
}

AofSBuff::~AofSBuff() {
    delete[] memoryPool;
}

//
//
//
void AofSBuff::AdvBuffers(){
    int t = previous;
    previous = current;
    current = next;
    next = t;
}


