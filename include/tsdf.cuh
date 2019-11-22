#ifndef _TSDF_CUH_
#define _TSDF_CUH_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vhashing.h"
#include "Utils.h"

#include <string>
#include <iostream>
#include <cmath>
#include <mutex>
#include "cutil_math.h"


#include <fstream>
#include <vector>
#include <list>
#include <cstdlib>
#include <unordered_map>
#include <cstring>
#include <GL/glew.h>
#include <GL/glut.h>
#include <utility>
#include <random>


#define T_PER_BLOCK 16

//minus
#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

//plus
#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

#define VOXEL_PER_BLOCK 6
// #define VOXEL_PER_BLOCK 6
#define BLOCK_PER_CHUNK 6
// #define BLOCK_PER_CHUNK 6
#define MAX_CPU2GPU_BLOCKS 10000
//#define MAX_CPU2GPU_BLOCKS 10000
#define MAX_CHUNK_NUM 64
// #define MAX_CHUNK_NUM 128
//#define CHUNK_RADIUS 3.0
#define CHUNK_RADIUS 2.0

#define MAXWEIGHT 20

__host__
static void FatalError(const int lineNumber = 0) {
    std::cerr << "FatalError";
    if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
    std::cerr << ". Program Terminated." << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

__host__
static void checkCUDA(const int lineNumber, cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
        FatalError();
    }
}

namespace ark {

    struct Vertex {
        float x;
        float y;
        float z;
        unsigned char r;
        unsigned char g;
        unsigned char b;

        __host__ __device__
        Vertex(){}
        __host__ __device__
        Vertex(float xi, float yi, float zi) : x(xi), y(yi), z(zi) {}
    };

    struct VertexEqual
    {
        bool operator()(Vertex v1, Vertex v2) const{
            return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
        }
        
    };

    struct VertexHasher
    {
        size_t operator()(Vertex v) const{
            const size_t p[] = {
                73856093,
                19349669,
                83492791
            };
            return ((size_t)v.x * p[0]) ^
                         ((size_t)v.y * p[1]) ^
                         ((size_t)v.z * p[2]);
        }
    };

    __host__ __device__
    bool operator==(const Vertex &a, const Vertex &b);

    struct Triangle {
        Vertex p[3];
        bool valid;

        Triangle():valid(false){}
    };

    struct Face {
        int vIdx[3];
    };

    struct GRIDCELL {
        Vertex p[8];
        float val[8];
    };

    struct Voxel {
        float sdf;
        unsigned char sdf_color[3];
        float weight;

        __device__ __host__
        Voxel():sdf(0), weight(0){
            sdf_color[0] = sdf_color[1] = sdf_color[2] = 0;
        }
    };

    struct VoxelBlock {
        Voxel voxels[VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK];
    };

    struct VoxelBlockPos{
        int3 pos;
        // int idx;
        __device__ __host__
        VoxelBlockPos():pos(make_int3(0,0,0)){};
    };

    bool operator==(const VoxelBlock &a, const VoxelBlock &b);

    struct BlockHasher {
        __device__ __host__
        size_t operator()(int3 patch) const {
            const size_t p[] = {
                73856093,
                19349669,
                83492791
            };
            return ((size_t)patch.x * p[0]) ^
                         ((size_t)patch.y * p[1]) ^
                         ((size_t)patch.z * p[2]);
        }
    };

    struct BlockEqual {
        __device__ __host__
        bool operator()(int3 patch1, int3 patch2) const {
            return patch1.x == patch2.x &&
                            patch1.y == patch2.y &&
                            patch1.z == patch2.z;
        }
    };

    struct Chunk{
        VoxelBlock *blocks;//一个chunk里面有多个block
        VoxelBlockPos *blocksPos;//与上面block类似 数组
        Triangle* tri_;
        bool isOccupied;

       __host__
        Chunk(): blocks(nullptr), blocksPos(nullptr), tri_(nullptr), isOccupied(false){}

        __host__
        void create(int3 pos){
            // printf("++create chunk at (%d,%d,%d)\n", pos.x, pos.y, pos.z);
            int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
            int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;

            if(blocks == nullptr){
                blocks = new VoxelBlock[block_total];
            }
            int3 block_start = pos * make_int3(BLOCK_PER_CHUNK);

            if(blocksPos == nullptr){
                blocksPos = new VoxelBlockPos[block_total];
                for(int x = 0; x < BLOCK_PER_CHUNK; x ++)
                    for(int y = 0; y < BLOCK_PER_CHUNK; y ++)
                        for(int z = 0; z < BLOCK_PER_CHUNK; z++){
                            blocksPos[(x * BLOCK_PER_CHUNK + y) * BLOCK_PER_CHUNK + z].pos = block_start + make_int3(x,y,z);
                        }
            }

            tri_ = new Triangle[total_vox * 5];// (Triangle *) malloc(sizeof(Triangle) * total_vox * 5);
        }

        __host__
        void release(){
            if(blocks == nullptr)
                return;

            // printf("--release chunk at\n");
            int3 startpos = blocksPos[0].pos;
            int3 pos = make_int3(startpos.x/ BLOCK_PER_CHUNK, startpos.y / BLOCK_PER_CHUNK, startpos.z/ BLOCK_PER_CHUNK);
            // printf(" (%d,%d,%d)\n", pos.x, pos.y, pos.z);

            delete[] blocks;
            delete[] blocksPos;
            delete[] tri_;
            isOccupied = false;

            blocks = nullptr;
            blocksPos = nullptr;
            tri_ = nullptr;
        }
    };

    class MarchingCubeParam {
    public:
        float3 vox_origin; // Location of voxel grid origin in base frame camera coordinates
        float vox_size;
        float trunc_margin;
        int3 vox_dim;
        int total_vox;
        float max_depth;
        float min_depth;
        float block_size;
        int im_width;
        int im_height;
        float fx, fy, cx, cy;

        int idxMap[8][3] = {{0, 0, 0},
                            {0, 1, 0},
                            {1, 1, 0},
                            {1, 0, 0},
                            {0, 0, 1},
                            {0, 1, 1},
                            {1, 1, 1},
                            {1, 0, 1}};

        int edgeTable[256] = {
                0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
                0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
                0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
                0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
                0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
                0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
                0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
                0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
                0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
                0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
                0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
                0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
                0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
                0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
                0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
                0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
                0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
                0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
                0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
                0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
                0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
                0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
                0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
                0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
                0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
                0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
                0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
                0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
                0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};


        int triTable[256][16] =
                {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  1,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  8,  3,  9,  8,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {2,  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
                 {3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  11, 2,  8,  11, 0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  9,  0,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1, -1, -1, -1},
                 {3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, -1},
                 {3,  9,  0,  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  4,  7,  3,  0,  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1},
                 {9,  2,  10, 9,  0,  2,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
                 {8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
                 {4,  7,  11, 9,  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, -1},
                 {3,  10, 1,  3,  11, 10, 7,  8,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  11, 10, 1,  4,  11, 1,  0,  4,  7,  11, 4,  -1, -1, -1, -1},
                 {4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
                 {4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, -1},
                 {9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  5,  4,  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  5,  4,  1,  5,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {8,  5,  4,  8,  3,  5,  3,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
                 {2,  10, 5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, -1},
                 {9,  5,  4,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  11, 2,  0,  8,  11, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
                 {2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1, -1, -1},
                 {10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {4,  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, -1},
                 {5,  4,  0,  5,  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
                 {5,  4,  8,  5,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1},
                 {9,  7,  8,  5,  7,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  7,  8,  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3,  -1, -1, -1, -1},
                 {8,  0,  2,  8,  2,  5,  8,  5,  7,  10, 5,  2,  -1, -1, -1, -1},
                 {2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, -1},
                 {2,  3,  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, -1},
                 {11, 2,  1,  11, 1,  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  5,  8,  8,  5,  7,  10, 1,  3,  10, 3,  11, -1, -1, -1, -1},
                 {5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,  10, 11, 10, 0,  -1},
                 {11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,  0,  -1},
                 {11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  0,  1,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  8,  3,  1,  9,  8,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, -1},
                 {2,  3,  11, 10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {11, 0,  8,  11, 2,  0,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1, -1},
                 {6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
                 {3,  11, 6,  0,  3,  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, -1},
                 {6,  5,  9,  6,  9,  11, 11, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1, -1, -1},
                 {1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
                 {6,  1,  2,  6,  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7,  -1, -1, -1, -1},
                 {8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6,  -1, -1, -1, -1},
                 {7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,  -1},
                 {3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, -1},
                 {0,  1,  9,  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1},
                 {9,  2,  1,  9,  11, 2,  9,  4,  11, 7,  11, 4,  5,  10, 6,  -1},
                 {8,  4,  7,  3,  11, 5,  3,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
                 {5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,  0,  4,  11, -1},
                 {0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,  -1},
                 {6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, -1},
                 {10, 4,  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  10, 6,  4,  9,  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 0,  1,  10, 6,  0,  6,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,  10, -1, -1, -1, -1},
                 {1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, -1},
                 {0,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {8,  3,  2,  8,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 4,  9,  10, 6,  4,  11, 2,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  -1, -1, -1, -1},
                 {3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1, -1, -1, -1},
                 {6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  -1},
                 {9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, -1},
                 {8,  11, 1,  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  -1},
                 {3,  11, 6,  3,  6,  0,  0,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {6,  4,  8,  11, 6,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1, -1, -1, -1, -1},
                 {0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1, -1},
                 {10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, -1},
                 {10, 6,  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, -1},
                 {2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9,  -1},
                 {7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
                 {7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, -1},
                 {2,  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  -1},
                 {1,  8,  0,  1,  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, -1},
                 {11, 2,  1,  11, 1,  7,  10, 6,  1,  6,  7,  1,  -1, -1, -1, -1},
                 {8,  9,  6,  8,  6,  7,  9,  1,  6,  11, 6,  3,  1,  3,  6,  -1},
                 {0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, -1},
                 {7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  0,  8,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  1,  9,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
                 {2,  9,  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
                 {6,  11, 7,  2,  10, 3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, -1},
                 {7,  2,  3,  6,  2,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, -1},
                 {10, 7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 7,  6,  1,  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, -1},
                 {0,  3,  7,  0,  7,  10, 0,  10, 9,  6,  10, 7,  -1, -1, -1, -1},
                 {7,  6,  10, 7,  10, 8,  8,  10, 9,  -1, -1, -1, -1, -1, -1, -1},
                 {6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  4,  6,  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, -1},
                 {6,  8,  4,  6,  11, 8,  2,  10, 1,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, 3,  0,  11, 0,  6,  11, 0,  4,  6,  -1, -1, -1, -1},
                 {4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,  -1, -1, -1, -1},
                 {10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,  -1},
                 {8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, -1},
                 {1,  9,  4,  1,  4,  2,  2,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10, 1,  -1, -1, -1, -1},
                 {10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
                 {4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  -1},
                 {10, 9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  9,  5,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  4,  9,  5,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  0,  1,  5,  4,  0,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
                 {11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1, -1, -1, -1},
                 {9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
                 {6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, -1},
                 {7,  6,  11, 5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, -1},
                 {3,  4,  8,  3,  5,  4,  3,  2,  5,  10, 5,  2,  11, 7,  6,  -1},
                 {7,  2,  3,  7,  6,  2,  5,  4,  9,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,  -1, -1, -1, -1},
                 {3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1, -1},
                 {6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  -1},
                 {9,  5,  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, -1},
                 {1,  6,  10, 1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  -1},
                 {4,  0,  10, 4,  10, 5,  0,  3,  10, 6,  10, 7,  3,  7,  10, -1},
                 {7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,  10, -1, -1, -1, -1},
                 {6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
                 {3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, -1},
                 {0,  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, -1},
                 {6,  11, 3,  6,  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  10, 9,  5,  11, 9,  11, 8,  11, 5,  6,  -1, -1, -1, -1},
                 {0,  11, 3,  0,  6,  11, 0,  9,  6,  5,  6,  9,  1,  2,  10, -1},
                 {11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,  2,  5,  -1},
                 {6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, -1},
                 {5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, -1},
                 {9,  5,  6,  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
                 {1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8,  -1},
                 {1,  5,  6,  2,  1,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,  8,  9,  6,  -1},
                 {10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1, -1},
                 {0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {10, 5,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {11, 5,  10, 7,  5,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {11, 5,  10, 11, 7,  5,  8,  3,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1, -1, -1},
                 {11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, -1},
                 {9,  7,  5,  9,  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, -1},
                 {7,  5,  2,  7,  2,  11, 5,  9,  2,  3,  2,  8,  9,  8,  2,  -1},
                 {2,  5,  10, 2,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1, -1, -1, -1},
                 {9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, -1},
                 {9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  -1},
                 {1,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  7,  0,  7,  1,  1,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  0,  3,  9,  3,  5,  5,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1},
                 {5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, -1},
                 {0,  1,  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, -1},
                 {10, 11, 4,  10, 4,  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  -1},
                 {2,  5,  1,  2,  8,  5,  2,  11, 8,  4,  5,  8,  -1, -1, -1, -1},
                 {0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11, 1,  5,  1,  11, -1},
                 {0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,  5,  -1},
                 {9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {2,  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, -1},
                 {5,  10, 2,  5,  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {3,  10, 2,  3,  5,  10, 3,  8,  5,  4,  5,  8,  0,  1,  9,  -1},
                 {5,  10, 2,  5,  2,  4,  1,  9,  2,  9,  4,  2,  -1, -1, -1, -1},
                 {8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
                 {0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, -1},
                 {9,  4,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  11, 7,  4,  9,  11, 9,  10, 11, -1, -1, -1, -1, -1, -1, -1},
                 {0,  8,  3,  4,  9,  7,  9,  11, 7,  9,  10, 11, -1, -1, -1, -1},
                 {1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11, -1, -1, -1, -1},
                 {3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,  -1},
                 {4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, -1},
                 {9,  7,  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  -1},
                 {11, 7,  4,  11, 4,  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
                 {11, 7,  4,  11, 4,  2,  8,  3,  4,  3,  2,  4,  -1, -1, -1, -1},
                 {2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,  9,  -1, -1, -1, -1},
                 {9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,  7,  -1},
                 {3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, -1},
                 {1,  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  9,  1,  4,  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
                 {4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1,  -1, -1, -1, -1},
                 {4,  0,  3,  7,  4,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, -1},
                 {0,  1,  10, 0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, -1},
                 {3,  1,  10, 11, 3,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  2,  11, 1,  11, 9,  9,  11, 8,  -1, -1, -1, -1, -1, -1, -1},
                 {3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,  -1, -1, -1, -1},
                 {0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {2,  3,  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
                 {9,  10, 2,  0,  9,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {2,  3,  8,  2,  8,  10, 0,  1,  8,  1,  10, 8,  -1, -1, -1, -1},
                 {1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {0,  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                 {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
    };

    __host__ __device__
    Vertex VertexInterp(const float& isolevel, const Vertex& p1, const Vertex& p2, const float& valp1, const float& valp2);

    // typedef struct 
    // {
    //     float tsdf;
    //     unsigned char tsdf_color;
    //     Triangle tri;
    //     float weight;
    // }Voxel;

    class GpuTsdfGenerator {
        int im_width_;
        int im_height_;

        // Voxel grid parameters (change these to change voxel grid resolution, etc.)
        float *TSDF_ = nullptr;
        unsigned char *TSDF_color_ = nullptr;
        float *weight_ = nullptr;
        MarchingCubeParam *param_;
        float K_[3 * 3];
        float c2w_[4 * 4];
	float w2c_[4*4];
        Triangle *tri_ = nullptr;
        Triangle* hash_tri_ = nullptr;
        float chunk_size;


        // Load variables to GPU memory

        float *dev_TSDF_ = nullptr;
        unsigned char *dev_TSDF_color_ = nullptr;
        float *dev_weight_ = nullptr;
        float *dev_K_ = nullptr;
        float *dev_c2w_ = nullptr;
        float *dev_depth_ = nullptr;
	float *dev_w2c_ = nullptr;
        unsigned char *dev_rgb_ = nullptr;
        Triangle *dev_tri_ = nullptr;
        MarchingCubeParam *dev_param_;

        //lock for rendering
        std::mutex tri_mutex_;
        //lock for saving ply
        std::mutex tsdf_mutex_;
        std::mutex chunk_mutex_;

        std::vector<Vertex> global_vertex;
        std::vector<Face> global_face;
        std::vector<std::list<std::pair<Vertex, int>>> global_map;


//hashing host
        Chunk* h_chunks;

        VoxelBlock* d_outBlock;
        VoxelBlockPos* d_outBlockPos;

        VoxelBlock* d_inBlock;
        VoxelBlockPos* d_inBlockPos;

        VoxelBlockPos* d_inBlockPosHeap;
        VoxelBlockPos* h_inBlockPosHeap;

        unsigned int *d_outBlockCounter;
        unsigned int *d_heapBlockCounter;
        unsigned int h_inChunkCounter;
        unsigned int h_heapBlockCounter;

    public:
        __host__
        GpuTsdfGenerator(int width, int height, float fx, float fy, float cx, float cy, float max_depth, float origin_x, float origin_y,
                         float origin_z, float vox_size, float trunc_m, int vox_dim_x, int vox_dim_y, int vox_dim__z);
	__host__
	void setMaxDepth(int depth);

        __host__
        void processFrame(float *depth, unsigned char *rgb, float *c2w, const Eigen::Vector4f&);

        __host__
        void getLocalGrid();

        __host__
        void insert_tri();

        __host__
        void render();

        __host__
        void Shutdown();

        __host__
        void SaveTSDF(std::string filename);

        __host__
        void SavePLY(std::string filename, const cv::Rect& foreground);

        __host__
        std::vector<Vertex>* getVertices();

        __host__
        std::vector<Face>* getFaces();

        __host__
        std::vector<std::list<std::pair<Vertex, int>>>* getHashMap();

        __host__
        MarchingCubeParam* getMarchingCubeParam();

        __host__
        int find_vertex(Vertex p, uint3 grid_size, float cell_size, std::vector<std::list<std::pair<Vertex, int>>>& hash_table);
    private:

        __host__
        void tsdf2mesh(std::string outputFileName, const cv::Rect& foreground);

        __host__
        int3 calc_cell_pos(Vertex p, float cell_size);

        __host__
        unsigned int calc_cell_hash(int3 cell_pos, uint3 grid_size);

        __host__
        void insert_vertex(Vertex p, int index, uint3 grid_size, float cell_size, std::vector<std::list<std::pair<Vertex, int>>>& hash_table);
    
        __host__ 
        void HashAssign(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w);

        __host__ 
        void HashReset();

        __host__
        bool isChunkInCameraFrustum(int x, int y, int z, float3 frustumCenter);

        __host__ bool isPosInCameraFrustum(float x, float y, float z);

        __host__ int chunkGetLinearIdx(int x, int  y, int z);

        __host__ void streamInCPU2GPU(float *K, float *c2w, float *depth);

        __host__ void streamOutGPU2CPU();

        __host__ int3 world2chunk(float3 pos);

        __host__ float3 getCameraPos();

        __host__ void clearheap();

        __host__ float3 getFrustumCenter(const float& depthValue, const float& xBias = 0, const float& yBias = 0);
    };

    // class TSDFHashMap
    // {
    //     thrust::host_vector<TSDF_tri*> data;
    //     size_t my_size;
    //     size_t capacity;

    // public:
    //     TSDFHashMap(size_t size);
    //     ~ TSDFHashMap();
    //     uint64_t getHash(int dim[3]);
    //     bool insert(int dim[3], TSDF_tri* TSDF_tri_);
    //     size_t size();
    // };


    // typedef struct 
    // {
    //     int position[3];
    //     int offset;
    //     Voxel* v;
    // }HashEntry;

}
#endif
