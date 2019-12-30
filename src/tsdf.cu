#include "tsdf.cuh"
#include <chrono>
#include <unordered_map>
#include <driver_types.h>
#include <device_launch_parameters.h>
//#include <cxcore.h>
#include <Eigen/src/Core/Matrix.h>
#include <Utils.h>
//当前设置的最大深度为10个单位 这个需要调整
//chunk的初始位置 也需要调整

// using namespace CUDASTL;
using std::vector;
using std::default_random_engine;
using namespace std::chrono;

#define DEBUG_WIDTH 32
#define DEBUG_HEIGHT 24
#define DEBUG_X  113
#define DEBUG_Y  26
#define DEBUG_Z  3
#define DDA_STEP 1 //DDA的步骤好像有点问题， 这个的大小代表每次在像素平面上的移动距离？
// CUDA kernel function to integrate a TSDF voxel volume given depth images
namespace ark {

    int plyCnt = 0;
    __host__
    void showPerFrame(Triangle * tri_, int triangleNum);

    __host__
    void tri2mesh(std::string outputFileName, Triangle* tri_, int triNum);

    // static const int LOCK_HASH = -1;
    //hashing device
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_chunks;
    vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>* dev_blockmap_;

    static  int countFrame = 0;

    __host__ void GpuTsdfGenerator::setMaxDepth(int depth) {
//        printf("!!!!!!!!!!!!!!\n");
        this->param_->max_depth = depth;
    }
    
    __device__ float3 voxel2world(int3 voxel, float voxel_size){
        float3 p;
        p.x = ((float)voxel.x) * voxel_size;
        p.y = ((float)voxel.y) * voxel_size;
        p.z = ((float)voxel.z) * voxel_size;
        return p;
    }

    __device__ int3 world2voxel(float3 voxelpos, float voxel_size){
        int3 p;
        p.x = floorf(voxelpos.x / voxel_size);
        p.y = floorf(voxelpos.y / voxel_size);
        p.z = floorf(voxelpos.z / voxel_size);
        return p;
    }

    __device__ float3 block2world(int3 idBlock, float block_size){
        float3 p;
        p.x = ((float)idBlock.x) * block_size;
        p.y = ((float)idBlock.y) * block_size;
        p.z = ((float)idBlock.z) * block_size;
        return p;
    }

    __device__ int3 wolrd2block(float3 blockpos, float block_size){
        int3 p;
        p.x = floorf(blockpos.x / block_size);
        p.y = floorf(blockpos.y / block_size);
        p.z = floorf(blockpos.z / block_size);
        return p;
    }

    __device__ int3 voxel2block(int3 idVoxel){
        return make_int3(floorf((float)idVoxel.x / (float)VOXEL_PER_BLOCK), floorf((float)idVoxel.y / (float)VOXEL_PER_BLOCK), floorf((float)idVoxel.z / (float)VOXEL_PER_BLOCK));
    }

    __device__ int voxelLinearInBlock(int3 idVoxel, int3 idBlock){
        int3 start_id = make_int3(idBlock.x * VOXEL_PER_BLOCK, idBlock.y * VOXEL_PER_BLOCK, idBlock.z * VOXEL_PER_BLOCK);
                    
        return ((idVoxel.x - start_id.x) * VOXEL_PER_BLOCK + (idVoxel.y - start_id.y))* VOXEL_PER_BLOCK + (idVoxel.z - start_id.z);
    }

    //camera function
    __host__    __device__
    void frame2cam(int* pt_pix, float pt_cam_z, float* pt_cam, float* K_){
        //convert bottom left of frame to current frame camera coordinates (camera)
        pt_cam[2] = pt_cam_z;
        pt_cam[0] = ((float)pt_pix[0] - K_[0 * 3 + 2]) * pt_cam_z / K_[0 * 3 + 0];
        pt_cam[1] = ((float)pt_pix[1] - K_[1 * 3 + 2]) * pt_cam_z / K_[1 * 3 + 1];

    }
    __host__ __device__
    void cam2frame(float* pt_cam, int *pt_pix, float* K){
        pt_pix[0] = roundf(K[0 * 3 + 0] * (pt_cam[0] / pt_cam[2]) + K[0 * 3 + 2]);
        pt_pix[1] = roundf(K[1 * 3 + 1] * (pt_cam[1] / pt_cam[2]) + K[1 * 3 + 2]);
        // printf("k0 is %f k2 is %f k4 is %f k5 is %f\n", K[0], K[2], K[4], K[5]);
        // printf("camera is %f %f %f frame is %d %d\n", pt_cam[0], pt_cam[1], pt_cam[2], pt_pix[0], pt_pix[1]);
    }

    __host__ __device__
    void base2cam(float* pt_base,float *pt_cam, float * c2w_){
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base[0] - c2w_[0 * 4 + 3];
        tmp_pt[1] = pt_base[1] - c2w_[1 * 4 + 3];
        tmp_pt[2] = pt_base[2] - c2w_[2 * 4 + 3];
        pt_cam[0] =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        pt_cam[1] =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        pt_cam[2] =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];
//        printf("base is %f %f %f\n camera is %f %f %f\n", pt_base[0], pt_base[1], pt_base[2],
//                pt_cam[0], pt_cam[1], pt_cam[2]);

    }

    __host__  __device__
    void cam2base(float *pt_cam, float* pt_base, float * c2w_){
        // Convert from current frame camera coordinates to base frame camera coordinates (wolrd)
        pt_base[0] = pt_cam[0] * c2w_[0 * 4 + 0] + pt_cam[1] * c2w_[0 * 4 + 1] + pt_cam[2] * c2w_[0 * 4 + 2] + c2w_[0 * 4 + 3];
        pt_base[1] = pt_cam[0] * c2w_[1 * 4 + 0] + pt_cam[1] * c2w_[1 * 4 + 1] + pt_cam[2] * c2w_[1 * 4 + 2] + c2w_[1 * 4 + 3];
        pt_base[2] = pt_cam[0] * c2w_[2 * 4 + 0] + pt_cam[1] * c2w_[2 * 4 + 1] + pt_cam[2] * c2w_[2 * 4 + 2] + c2w_[2 * 4 + 3];
    }

    __host__  __device__
    float3 frame2base(const int& pt_pix_x, const int& pt_pix_y,
                const float& pt_cam_z, float* K_, float * c2w_, MarchingCubeParam *param){
        float pt_cam[3];
        int pt_pix[2] = {pt_pix_x, pt_pix_y};
        frame2cam(pt_pix, pt_cam_z, pt_cam, K_);
        float pt_base[3];
        cam2base(pt_cam, pt_base, c2w_);
        // 显示从像素坐标系到世界坐标系的转换。验证成功。
        // if(pt_cam_z && pt_cam_z != 10.0) {
        //      printf("camera is %f %f %f\n base is %f %f %f\n pt is %d %d %f\n"
        //      ,pt_cam[0],pt_cam[1], pt_cam[2], pt_base[0], pt_base[1], pt_base[2], pt_pix_x, pt_pix_y, pt_cam_z);
        // }

        return make_float3(pt_base[0], pt_base[1], pt_base[2]);
    }

    __host__ bool GpuTsdfGenerator::isPosInCameraFrustum(float x, float y, float z){
        float pt_base_x = x;// param_->vox_origin.x + x;
        float pt_base_y = y;//param_->vox_origin.y + y;
        float pt_base_z = z;//param_->vox_origin.z + z;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - c2w_[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - c2w_[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - c2w_[2 * 4 + 3];
        float pt_cam_x =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
            return false;

        if(pt_cam_z > param_->max_depth)
            return false;

        int pt_pix_x = roundf(K_[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K_[0 * 3 + 2]);
        int pt_pix_y = roundf(K_[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K_[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width_ || pt_pix_y < 0 || pt_pix_y >= im_height_)
            return false;

        return true;
    }

    __host__ float3 GpuTsdfGenerator::getCameraPos(){
        // return make_float3(c2w_[0 * 4 + 3] - param_->vox_origin.x,c2w_[1 * 4 + 3]- param_->vox_origin.y,c2w_[2 * 4 + 3]- param_->vox_origin.z);
        return make_float3(c2w_[0 * 4 + 3],c2w_[1 * 4 + 3],c2w_[2 * 4 + 3]);
    }

    __host__ float3 GpuTsdfGenerator::getFrustumCenter(const float &depthValue,
                                                const float& xBias, const float& yBias){
        // float3 getCameraPos();
        int x = im_width_ / 2 + xBias;
        //修改了流的中心位置。
        int y = im_height_ / 2 + yBias;

        float d = depthValue;

//        std::cout<< "k is "<< c2w_[0]<< std::endl;

        return std::move(frame2base(x,y,d, K_, c2w_, param_));
        // printf("raymin\t%f\t%f\t%f\traymax \t%f\t%f\t%f\torigin\t%f\t%f\t%f\n", 
        //         raymin.x, raymin.y, raymin.z, raymax.x, raymax.y, raymax.z, camcenter.x, camcenter.y, camcenter.z);
    }

    __host__ bool GpuTsdfGenerator::isChunkInCameraFrustum(int x, int y, int z, float3 frustumCenter){

        float3 chunk_center = make_float3(((float)x + 0.5) * chunk_size, ((float)y + 0.5) * chunk_size, ((float)z + 0.5) * chunk_size);
        // float3 cam_center = getCameraPos();
        // printf("host: isChunkInCameraFrustum  %f %f %f\n", chunk_center.x, chunk_center.y, chunk_center.z);
        // 之前的chunk_radius定义方法
        // float chunkRadius = 0.5f*CHUNK_RADIUS*sqrt(3.0f) * 1.1;
        //当前将chunk半径直接定为3
        float  chunkRadius = CHUNK_RADIUS * chunk_size;
        float3 vec = (frustumCenter - chunk_center);
        float l = sqrt(vec.x * vec.x + vec.z * vec.z + vec.y * vec.y);
        // printf("\t\t chunkRadius %f \t distance %f\n", chunkRadius, l);
        // printf("distance is %f\n chunk radius is %f\n", l, chunkRadius);
        if(l <= std::abs(chunkRadius))
            return true;
        else
            return false;
    }

    __host__ int GpuTsdfGenerator::chunkGetLinearIdx(int x, int  y, int z){
        int dimx = x + MAX_CHUNK_NUM / 2;
        int dimy = y + MAX_CHUNK_NUM / 2;
        int dimz = z + MAX_CHUNK_NUM / 2;

        return (dimx * MAX_CHUNK_NUM + dimy) * MAX_CHUNK_NUM + dimz;
    }

    __host__ int3 GpuTsdfGenerator::world2chunk(float3 pos){
        int3 p;
        // chunk_size: 每个chunk的具体长度
        p.x = floor(pos.x/chunk_size);
        p.y = floor(pos.y/chunk_size);
        p.z = floor(pos.z/chunk_size);
        return std::move(p);
    }

    __global__ void streamInCPU2GPUKernel(int h_inChunkCounter, VoxelBlock* d_inBlock, VoxelBlockPos* d_inBlockPos, vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks){
        const unsigned int bucketId = blockIdx.x * blockDim.x + threadIdx.x;
        // const uint total_vx_p_block = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
        if(bucketId < h_inChunkCounter){
            int3 pos = d_inBlockPos[bucketId].pos;
            // printf("Pos is %d %d %d\n",pos.x,pos.y,pos.z);
            // if(dev_blockmap_chunks.find(pos) == dev_blockmap_chunks.end())
            //关键步骤 获取位置 插入到hash列表当中 涉及到hash方法
            dev_blockmap_chunks[pos] = d_inBlock[bucketId];
        }
    }

    // 开个线程检查分配方案是否正确
    __global__ void checkStreamInKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks){
        const unsigned int bucketId = blockIdx.x * blockDim.x + threadIdx.x;

            if(bucketId == 0){
                if(dev_blockmap_chunks.find(make_int3(999,999,999)) != dev_blockmap_chunks.end())
                    printf("insert successfully  (999,999,999)\n");
                else
                    printf("not 999 999 999\n");

                VoxelBlock vb;
                dev_blockmap_chunks[make_int3(999,999,999)] = vb;

                if(dev_blockmap_chunks.find(make_int3(999,999,999)) != dev_blockmap_chunks.end())
                    printf("insert successfully  (999,999,999)\n");
                else
                    printf("not 999 999 999\n");
            }
    }

    __host__ int3 blockLinear2Int3(int idLinearBlock){
        int z = idLinearBlock % BLOCK_PER_CHUNK;
        int y = floor((idLinearBlock % (BLOCK_PER_CHUNK * BLOCK_PER_CHUNK))/BLOCK_PER_CHUNK);
        int x = floor(idLinearBlock / (BLOCK_PER_CHUNK * BLOCK_PER_CHUNK));
        return make_int3(x,y,z);
    }


    __host__ int3 chunk2block(int3 idChunk, int idLinearBlock){
        int3 id = blockLinear2Int3(idLinearBlock);

        int x = idChunk.x * BLOCK_PER_CHUNK + id.x;
        int y = idChunk.y * BLOCK_PER_CHUNK + id.y;
        int z = idChunk.z * BLOCK_PER_CHUNK + id.z;
        return make_int3(x,y,z);
    }

    __host__ __device__ int3 block2chunk(int3 idBlock){
        return make_int3(floor((float)idBlock.x/(float)BLOCK_PER_CHUNK),
            floor((float)idBlock.y/(float)BLOCK_PER_CHUNK),
            floor((float)idBlock.z/(float)BLOCK_PER_CHUNK));
    }

    __host__ int blockPos2Linear(int3 blocksPos){
        int3 chunkpos = block2chunk(blocksPos);
        int3 insideId = blocksPos - chunkpos * make_int3(BLOCK_PER_CHUNK);
        return (insideId.x * BLOCK_PER_CHUNK + insideId.y) * BLOCK_PER_CHUNK + insideId.z;
    }

    __host__ float3 chunk2world(int3 idChunk, float chunk_size){
        return make_float3(((float)(idChunk.x) + 0.5)* chunk_size,
            ((float)(idChunk.y) + 0.5)* chunk_size,
            ((float)(idChunk.z) + 0.5)* chunk_size);
    }

    __host__ void GpuTsdfGenerator::streamInCPU2GPU(float *K, float *c2w, float *depth){


        auto a = system_clock::now();
        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        h_inChunkCounter = 0;

        // printf("HOST: streamInCPU2GPU\n");

//        float3 cam_pos = getCameraPos();
//        printf("cam_pos = %f %f %f\n", cam_pos.x, cam_pos.y, cam_pos.z);

        float3 frustumCenter = getFrustumCenter(700, 0, 0);
//        float3 frustumCenter = getFrustumCenter(600, 0, 100);
        printf("frustumCenter = %f %f %f\n", frustumCenter.x, frustumCenter.y, frustumCenter.z);
        auto b = system_clock::now();
        ark::printTime(a,b, "计算流中心");

        a = system_clock::now();
        int3 camera_chunk = world2chunk(frustumCenter);
        printf("camera_chunk = %d %d %d \n", camera_chunk.x, camera_chunk.y, camera_chunk.z);
        b = system_clock::now();
        ark::printTime(a, b, "摄像头chunk的位置");
        // float3 chunk_center = chunk2world(camera_chunk, chunk_size);
        // printf("MAX_CHUNK_NUM = %d chunk start from %d of size %f\n", MAX_CHUNK_NUM, -MAX_CHUNK_NUM/2, chunk_size);
        // printf("frustum_chunk = %d %d %d  pos = %f %f %f block = %d %d %d \n", camera_chunk.x, camera_chunk.y, camera_chunk.z,
        //     chunk_center.x, chunk_center.y, chunk_center.z,
        //     camera_chunk.x * BLOCK_PER_CHUNK, camera_chunk.y * BLOCK_PER_CHUNK, camera_chunk.z * BLOCK_PER_CHUNK);
        // 之前的chunk范围
        // int chunk_range_i = ceil(CHUNK_RADIUS/chunk_size);
        int chunk_range_i = CHUNK_RADIUS;
        printf("chunk_range_i = %d\n", chunk_range_i);
        a = system_clock::now();

        int3 chunk_start = make_int3(max(camera_chunk.x - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.y - chunk_range_i, - MAX_CHUNK_NUM / 2),
            max(camera_chunk.z - chunk_range_i, - MAX_CHUNK_NUM / 2));

        printf("chunk_start = %d %d %d\n", chunk_start.x, chunk_start.y, chunk_start.z);

        int3 chunk_end = make_int3(min(camera_chunk.x + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.y + chunk_range_i, MAX_CHUNK_NUM / 2 - 1),
            min(camera_chunk.z + chunk_range_i, MAX_CHUNK_NUM / 2 - 1));

        printf("chunk_end = %d %d %d\n", chunk_end.x, chunk_end.y, chunk_end.z);
        b = system_clock::now();
        ark::printTime(a, b, "chunk start end Time");
//        a = system_clock::now();
        {
            // unique_lock 加锁操作。
            std::unique_lock<std::mutex> lock(chunk_mutex_);
//        b = system_clock::now();
//        ark::printTime(a,b,"lock");

            a = system_clock::now();
            //x y z  real pos idx
            for (int x = chunk_start.x; x <= chunk_end.x; x++) {
                for (int y = chunk_start.y; y <= chunk_end.y; y++) {
                    for (int z = chunk_start.z; z <= chunk_end.z; z++) {
                        int idChunk = chunkGetLinearIdx(x, y, z);

                        if (isChunkInCameraFrustum(x, y, z, frustumCenter)) {
                            if (h_chunks[idChunk].blocks == nullptr) {
                                h_chunks[idChunk].create(make_int3(x, y, z));
                            }
//                        printf("cuda malloc total %d  max :% d\n", h_inChunkCounter, MAX_CPU2GPU_BLOCKS);
                            cudaSafeCall(cudaMemcpy(d_inBlock + h_inChunkCounter, h_chunks[idChunk].blocks,
                                                    sizeof(VoxelBlock) * block_total, cudaMemcpyHostToDevice));
                            cudaSafeCall(cudaMemcpy(d_inBlockPos + h_inChunkCounter, h_chunks[idChunk].blocksPos,
                                                    sizeof(VoxelBlockPos) * block_total, cudaMemcpyHostToDevice));
                            // h_chunks[idChunk].isOnGPU = 1;
                            h_inChunkCounter += block_total;
                        } else if (h_chunks[idChunk].isOccupied == false) {
                            // 因为render原因将一些chunk删除掉，所以导致了空洞的产生。
                            h_chunks[idChunk].release();
                        }
                    }
                }
            }
        }

        b = system_clock::now();
        ark::printTime(a, b, "COPY H_CHUNKS 2 d_inBlockPOS");

        a  = system_clock::now();

        if(h_inChunkCounter > 0){
            const dim3 grid_size((h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
            const dim3 block_size(T_PER_BLOCK * T_PER_BLOCK, 1);
            printf("grid is %d, block is %d\n",
                    (h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK),
                    T_PER_BLOCK * T_PER_BLOCK);

            // printf(" streamInCPU2GPUKernel grid %d  block %d \n", (h_inChunkCounter + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), T_PER_BLOCK * T_PER_BLOCK);
            // VoxelBlockPos *h_inBlockPos;
            // h_inBlockPos = (VoxelBlockPos *)malloc(sizeof(VoxelBlockPos) * h_inChunkCounter);
            // cudaSafeCall(cudaMemcpy(h_inBlockPos, d_inBlockPos, sizeof(VoxelBlockPos) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            // VoxelBlock* h_inBlock;
            // h_inBlock = (VoxelBlock *)malloc(sizeof(VoxelBlock) * h_inChunkCounter);
            // cudaSafeCall(cudaMemcpy(h_inBlock, d_inBlock, sizeof(h_inBlock) * h_inChunkCounter, cudaMemcpyDeviceToHost));

            // for(int i = 0; i < h_inChunkCounter; i += 1000){
            //     printf("blockpos[%d] at %d %d %d\n", i, h_inBlockPos[i].pos.x,h_inBlockPos[i].pos.y,h_inBlockPos[i].pos.z);
            //     printf("block[%d] of %f\n", i, h_inBlock[i].voxels[0].sdf);
            // }

            streamInCPU2GPUKernel<<<grid_size, block_size >>> (h_inChunkCounter, d_inBlock, d_inBlockPos, *dev_blockmap_chunks);
            // return;
            // cudaSafeCall(cudaDeviceSynchronize()); //debug

            // stream in
            // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
            // bmhi(*dev_blockmap_chunks);
            // int count = 0;
            // for(int i = 0; i < *(bmhi.heap_counter); i ++){
            //     for(int j = 0; j < 4; j ++){
            //         int offset = i * 4 + j;
            //         vhashing::HashEntryBase<int3> &entr = bmhi.hash_table[offset];
            //         int3 ii = entr.key;
            //         if(ii.x == 999999 && ii.y == 999999 && ii.z == 999999 )
            //             continue;
            //         else{
            //             count ++;
            //         }
            //     }
            // }
            // printf("* get heap_counter %d \n", *(bmhi.heap_counter));

            // int recount = 0;
            // // stream in
            // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
            // bmhhh(100001, 4, 400000, int3{999999, 999999, 999999});

            // // check
            // for (int i=0; i<*(bmhi.heap_counter); i++) {
            //     int3 key = bmhi.key_heap[i];
            //     // printf("key\t%d\t%d\t%d\n", key.x, key.y, key.z);
            //     if(bmhhh.find(key) == bmhhh.end()){
            //         VoxelBlock vb;
            //         bmhhh[key] = vb;
            //         recount ++;
            //     }
            // }

            // printf("count out ==== %d\n", recount);
            // for(int i = 0; i < h_inChunkCounter; i ++){
            //     VoxelBlock vb = bmhi[h_inBlockPos[i].pos];
            //     printf("pos = %d %d %d \n", h_inBlockPos[i].pos.x, h_inBlockPos[i].pos.y, h_inBlockPos[i].pos.z);
            // }
        }
        b = system_clock::now();
        ark::printTime(a, b, "device copy");
    }

    __global__ void getMapValueKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
        VoxelBlock* d_outBlock, VoxelBlockPos* d_outBlockPosHeap){
        const unsigned int idheap = blockIdx.x * blockDim.x + threadIdx.x;
        if(idheap < *(dev_blockmap_.heap_counter)){
            int3 pos = dev_blockmap_.key_heap[idheap];
            d_outBlockPosHeap[idheap].pos = pos;
            d_outBlock[idheap] = dev_blockmap_[pos];
        }
    }

    __host__ void GpuTsdfGenerator::streamOutGPU2CPU(){

        VoxelBlock* d_outBlock;
        VoxelBlockPos* d_outBlockPosHeap;

        cudaSafeCall(cudaMalloc(&d_outBlock, sizeof(VoxelBlock) * h_heapBlockCounter));
        cudaSafeCall(cudaMalloc(&d_outBlockPosHeap, sizeof(VoxelBlockPos) * h_heapBlockCounter));
        {
            const dim3 grid_size((h_heapBlockCounter + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
            const dim3 block_size(T_PER_BLOCK, 1);
            getMapValueKernel <<< grid_size, block_size >>> (*dev_blockmap_, d_outBlock, d_outBlockPosHeap);
        }
        VoxelBlock* h_outBlock;
        VoxelBlockPos* h_outBlockPosHeap;

        h_outBlock = new VoxelBlock[h_heapBlockCounter];
        h_outBlockPosHeap = new VoxelBlockPos[h_heapBlockCounter];

        cudaSafeCall(cudaMemcpy(h_outBlock, d_outBlock, sizeof(VoxelBlock) * h_heapBlockCounter, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(h_outBlockPosHeap, d_outBlockPosHeap, sizeof(VoxelBlockPos) * h_heapBlockCounter, cudaMemcpyDeviceToHost));

        cudaFree(d_outBlock);
        cudaFree(d_outBlockPosHeap);

        std::unique_lock<std::mutex> lock(tri_mutex_);
        std::unique_lock<std::mutex> lockChunk(chunk_mutex_);

        int counter = h_heapBlockCounter;
        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
        for(int i = 0; i < counter; i++){
            int3 pos = h_outBlockPosHeap[i].pos;
            VoxelBlock& vb = h_outBlock[i];

            int3 startpos = pos * make_int3(VOXEL_PER_BLOCK);
            int3 chunkpos = block2chunk(pos);
            int idChunk = chunkGetLinearIdx(chunkpos.x, chunkpos.y, chunkpos.z);
            h_chunks[idChunk].isOccupied = true;

            int idBlock = blockPos2Linear(pos);
//            printf("idBlock is %d\n", idBlock);
            int3 iddblock = blockLinear2Int3(idBlock) + chunkpos * BLOCK_PER_CHUNK;

//             printf("(%d,%d,%d) and (%d,%d,%d)\n", iddblock.x, iddblock.y, iddblock.z, pos.x, pos.y, pos.z);
            // assert(iddblock.x == pos.x && iddblock.y == pos.y && iddblock.z == pos.z);

            memcpy(&(h_chunks[idChunk].blocks[idBlock]), &vb, sizeof(Voxel) * total_vox);
            for(int x = 0; x< total_vox; x++) {
                Triangle* tri_src = &hash_tri_[i * total_vox * 5 + x * 5];

                Triangle* tri_dst = &(h_chunks[idChunk].tri_[idBlock * total_vox * 5 + x *5]);

                if(tri_src->valid)
                    memcpy(tri_dst, tri_src, sizeof(Triangle) * 5);

            }


            // printf("%f copy to %f\n", h_chunks[idChunk].blocks->voxels[5].sdf, vb.voxels[5].sdf);
            // assert(h_chunks[idChunk].blocks->voxels[0].sdf == vb.voxels[0].sdf);
        };
        free(hash_tri_);
        delete[] h_outBlock;
        delete[] h_outBlockPosHeap;
        // cudaSafeCall(cudaDeviceSynchronize()); //debug
        // std::ofstream outFile;
        // outFile.open("triangles_chunk"+ std::to_string(countFrame) + ".txt");

        // {
        //     int count = 0;
        //     int chunk_half = MAX_CHUNK_NUM / 2;
        //     int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        //     int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        //     int tri_num = total_vox * 5;
        //     int empty = 0;
        //     for(int x = - chunk_half; x < chunk_half; x ++){
        //         for(int y = - chunk_half; y < chunk_half; y ++){
        //             for(int z = - chunk_half; z < chunk_half; z ++){
        //                 int id = chunkGetLinearIdx(x,y,z);
        //                 Triangle* tri_ = h_chunks[id].tri_;
        //                 if(tri_ != nullptr){
        //                     outFile << "-----------\t"<< x << "\t" <<y<<"\t"<<z<<"\t"<<"-----------\n";
        //                     int count = 0;
        //                     for(int i = 0; i < tri_num; i ++){
        //                         if(!h_chunks[id].tri_[i].valid)
        //                             continue;
        //                         for (int j = 0; j < 3; ++j){
        //                             outFile << tri_[i].p[j].x << "\t" << tri_[i].p[j].y << "\t" << tri_[i].p[j].z << "\t";
        //                         }
        //                         outFile << "\n";
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // outFile.close();

        // delete h_blockmap_;

    }


    __global__ void IntegrateHashKernel(float *K, float *c2w, float *depth, unsigned char *rgb,
                   int height, int width, MarchingCubeParam *param,  
                   vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
                   VoxelBlockPos* d_inBlockPosHeap, unsigned int *d_heapBlockCounter,
                   float* planeParam){

        unsigned int idheap = blockIdx.x;
        if(idheap < *d_heapBlockCounter){
//            printf("planeparam is %f %f %f %f\n", planeParam[0], planeParam[1], planeParam[2], planeParam[3]);
            int3 idBlock = dev_blockmap_.key_heap[idheap];
            VoxelBlock& vb = dev_blockmap_[idBlock];
            Voxel& voxel = vb.voxels[threadIdx.x];

            int VOXEL_PER_BLOCK2 = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
            int z = threadIdx.x % VOXEL_PER_BLOCK;
            int y = ((threadIdx.x - z) % VOXEL_PER_BLOCK2) / VOXEL_PER_BLOCK;
            int x = threadIdx.x / VOXEL_PER_BLOCK2;
//            if (idheap == 20000)
//                printf("block\t%d\tthread\t%d\trank\t%d\tvoxel\t(%d,%d,%d)\n", blockIdx.x, threadIdx.x, blockIdx.x*blockDim.x+threadIdx.x, x, y, z);

            int3 idVoxel = idBlock * VOXEL_PER_BLOCK + make_int3(x,y,z);
            float3 voxelpos = voxel2world(idVoxel, param->vox_size);

            //此处添加一个形参：float4 表征平面法向量。
//            printf("%f %f %f %f \n", planeParam[0], planeParam[1], planeParam[2], planeParam[3]);


            // Convert from base frame camera coordinates to current frame camera coordinates
            float pt_base[3] = {voxelpos.x,// + param->vox_origin.x,
                                voxelpos.y,// + param->vox_origin.y,
                                voxelpos.z};// + param->vox_origin.z};// voxel2world(idVoxel, param->vox_size);

            float pt_cam[3] = {0};
            // printf("ptCam is %f %f %f\n", pt_cam[0], pt_cam[1], pt_cam[2]);

            base2cam(pt_base, pt_cam, c2w);
            float  pt_cam_z = pt_cam[2];

            //由于RT不一定准确所以暂时停止这个方法，直接在深度图上进行处理。
//            if(fabs(pt_cam[0] * planeParam[0] + planeParam[1] * pt_cam[1] +
//                    planeParam[2] * pt_cam[2] + planeParam[3]) < 10)
//                return;

            int pt_pix[2];
            cam2frame(pt_cam, pt_pix, K);

            int pt_pix_x = pt_pix[0];
            int pt_pix_y = pt_pix[1];

            if (pt_cam_z <= 0)
                return;

            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                return;



            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            //当前发现出现边缘垃圾的原因是 去除地面过程中造成的误差。
//            if(voxelpos.z > 800)
//                printf(" 800 %d %d %f %f\n", pt_pix_x, pt_pix_y, pt_cam_z, depth_val);

            // printf("depth_pixel  is %f depth_value is %f\n", depth_val, pt_cam_z);
            if (depth_val <= 0 || depth_val > param->max_depth)
                return;
            //TODO::genglishuai 此处的计算公式并不是直接相减（详细的内容还需要看相关论文）
//            float diff = depth_val - sqrt(pt_cam[0] *  pt_cam[0]
//                    + pt_cam[1] *  pt_cam[1] +
//                    pt_cam[2] *  pt_cam[2]);
            float diff = depth_val - pt_cam_z;



//            if(fabs(diff) > 20)
//                return;

            int image_idx = pt_pix_y * width + pt_pix_x;
            float dist = 0;
            if (diff >= 0)
                dist = fmin(1.0f, diff*1.0 / param->trunc_margin);
            else
                dist = fmax(-1.0f, diff*1.0 / param->trunc_margin);

            if(dist + 1.0f < 0.0001)
                return;

            float weight_old = voxel.weight;
            float weight_new = ((weight_old + 1.0f) < (MAXWEIGHT)) ? (weight_old + 1.0f) : MAXWEIGHT;
//            (weight_old && dist * voxel.sdf < 0 && fabs(dist) > 0.99)
//            if (weight_new > MAXWEIGHT) {
//                return;
//            }
            //设定当前的权重范围，如果超过最大权重则不再进行迭代更新。

//            if (weight_new > MAXWEIGHT)
//                weight_new = MAXWEIGHT;

            voxel.weight = weight_new;
            // 查看体素的权重是否有被更新。
            // printf("Tsdf is %f\n", voxel.weight);
            float oldSdf = voxel.sdf;
            voxel.sdf = (voxel.sdf * weight_old + dist * weight_new) * 1.0 /(weight_new + weight_old);
//            if(voxel.sdf * oldSdf < 0) {
//                printf(" idvoxel is %d %d %d currentSDF is%f  oldSDF is %f  diff is%f  weight is%f\n",idVoxel.x, idVoxel.y, idVoxel.z,
//                       voxel.sdf, oldSdf, diff, voxel.weight);
//            }


            // 当前存在上色过程中模糊化的问题，间隔上色并不能弥补该问题。
            if(voxel.sdf_color[0] < 0.0001 && voxel.sdf_color[1] < 0.0001 && voxel.sdf_color[2] < 0.0001) {
                voxel.sdf_color[0] = (voxel.sdf_color[0] * weight_old + rgb[3 * image_idx] * weight_new) / (weight_new + weight_old);
                voxel.sdf_color[1] = (voxel.sdf_color[1] * weight_old + rgb[3 * image_idx + 1] * weight_new) / (weight_new + weight_old);
                voxel.sdf_color[2] = (voxel.sdf_color[2] * weight_old + rgb[3 * image_idx + 2] * weight_new) / (weight_new + weight_old);
            }
            //            voxel.sdf_color[0] = rgb[3 * image_idx];
//            voxel.sdf_color[1] = rgb[3 * image_idx + 1];
//            voxel.sdf_color[2] = rgb[3 * image_idx + 2];

//            if(voxel.sdf_color[0] == 0 && voxel.sdf_color[1] == 0 && voxel.sdf_color[2] == 0){
//                printf("voxel sdf is %f %d\n", voxel.sdf, voxel.weight);
//            }

//            if ((int)voxel.weight % 3 == 0) {
//                voxel.sdf_color[0] = (voxel.sdf_color[0] * weight_old + rgb[3 * image_idx]) / weight_new;
//                voxel.sdf_color[1] = (voxel.sdf_color[1] * weight_old + rgb[3 * image_idx + 1]) / weight_new;
//                voxel.sdf_color[2] = (voxel.sdf_color[2] * weight_old + rgb[3 * image_idx + 2]) / weight_new;
//            }

        }
    }
    __global__
    void DropGarbageHashKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
                                        unsigned int *d_heapBlockCounter) {
        printf("%d %d\n", blockIdx.x, threadIdx.x);
        unsigned int idheap = blockIdx.x;
        if(idheap < *d_heapBlockCounter) {
            printf("进入\n");

            int3 idBlock = dev_blockmap_.key_heap[idheap];
            VoxelBlock& vb = dev_blockmap_[idBlock];
            unsigned int weight = 0;
            float tsdf = 0;

            for(int i = 0 ; i <VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK; i++) {

                Voxel& voxel = vb.voxels[i];
                if( voxel.weight > weight)
                    weight = voxel.weight;
                float tem = fabs(voxel.sdf);
                if( tem > tsdf)
                    tsdf = tem;
            }

            if(!weight || tsdf > 40)
                dev_blockmap_.erase(idBlock);
        }

    }

    __host__
    void DropGarbage(unsigned int h_heapBlockCounter, unsigned int *d_heapBlockCounter){

        printf("进入1\n");
        //进行Garbage收集
        const dim3 gridSize(h_heapBlockCounter, 1);
        const dim3 blockSize(1, 1);

        DropGarbageHashKernel <<< gridSize, blockSize>>> (*dev_blockmap_, d_heapBlockCounter);

        //垃圾收集结束后更新d_heapBlockCount;

        *d_heapBlockCounter = *(dev_blockmap_->heap_counter);

    }

    __global__
    void Integrate(float *K, float *c2w, float *depth, unsigned char *rgb,
                   int height, int width, float *TSDF, unsigned char *TSDF_color,
                   float *weight, MarchingCubeParam *param) {

        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        for (int pt_grid_x = 0; pt_grid_x < param->vox_dim.x; ++pt_grid_x) {

            // Convert voxel center from grid coordinates to base frame camera coordinates
            float pt_base[3];
            // pt_base[0] = pt_grid_x * param->vox_size;//param->vox_origin.x + pt_grid_x * param->vox_size;
            // pt_base[1] = pt_grid_y * param->vox_size;//param->vox_origin.y + pt_grid_y * param->vox_size;
            // pt_base[2] = pt_grid_z * param->vox_size;//param->vox_origin.z + pt_grid_z * param->vox_size;
            pt_base[0] = param->vox_origin.x + pt_grid_x * param->vox_size;
            pt_base[1] = param->vox_origin.y + pt_grid_y * param->vox_size;
            pt_base[2] = param->vox_origin.z + pt_grid_z * param->vox_size;

            // Convert from base frame camera coordinates to current frame camera coordinates
            float tmp_pt[3] = {0};
            tmp_pt[0] = pt_base[0] - c2w[0 * 4 + 3];
            tmp_pt[1] = pt_base[1] - c2w[1 * 4 + 3];
            tmp_pt[2] = pt_base[2] - c2w[2 * 4 + 3];
            float pt_cam_x =
                    c2w[0 * 4 + 0] * tmp_pt[0] + c2w[1 * 4 + 0] * tmp_pt[1] + c2w[2 * 4 + 0] * tmp_pt[2];
            float pt_cam_y =
                    c2w[0 * 4 + 1] * tmp_pt[0] + c2w[1 * 4 + 1] * tmp_pt[1] + c2w[2 * 4 + 1] * tmp_pt[2];
            float pt_cam_z =
                    c2w[0 * 4 + 2] * tmp_pt[0] + c2w[1 * 4 + 2] * tmp_pt[1] + c2w[2 * 4 + 2] * tmp_pt[2];
            float pt_cam[3] = {pt_cam_x, pt_cam_y, pt_cam_z};

            base2cam(pt_base, pt_cam, c2w);
            pt_cam_x = pt_cam[0];
            pt_cam_y = pt_cam[1];
            pt_cam_z = pt_cam[2];

            int pt_pix[2];
            cam2frame(pt_cam, pt_pix, K);

            int pt_pix_x = pt_pix[0];//roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            int pt_pix_y = pt_pix[1];//roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);

            if(pt_grid_z == 10 && pt_grid_y == 20 && pt_grid_x == 30){
                float cam_pos[3];
                frame2cam(pt_pix, pt_cam_z, cam_pos, K);
                printf("%f %f %f ==== %f %f %f \n", cam_pos[0], cam_pos[1], cam_pos[2], pt_cam[0], pt_cam[1], pt_cam[2]);
                float base_pos[3];
                cam2base(cam_pos, base_pos, c2w);
                printf("%f %f %f ==== %f %f %f \n", base_pos[0], base_pos[1], base_pos[2], pt_base[0], pt_base[1], pt_base[2]);
            }


            if (pt_cam_z <= 0)
                continue;

            // int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
            // int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
            if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
                continue;

            float depth_val = depth[pt_pix_y * width + pt_pix_x];

            if (depth_val <= 0 || depth_val > param->max_depth)
                continue;

            float diff = depth_val - pt_cam_z;

            if (diff <= -param->trunc_margin)
                continue;

            if(pt_pix_x % DEBUG_WIDTH == 0 && pt_pix_y % DEBUG_HEIGHT == 0){
                // int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
                // int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
                float depth_val = depth[pt_pix_y * width + pt_pix_x];
                float diff = depth_val - pt_cam_z;
                int3 idx3 = world2voxel(make_float3((float)pt_base[0], (float)pt_base[1], (float)pt_base[2]), param->vox_size);
                int3 idBlock = voxel2block(idx3);
                printf("good pos\t(%f,%f,%f)\tblock\t(%d,%d,%d)\tvoxel\t(%d,%d,%d)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
                pt_base[0], pt_base[1], pt_base[2], 
                idBlock.x, idBlock.y, idBlock.z,
                idx3.x, idx3.y, idx3.z,
                pt_cam_x, pt_cam_y, pt_cam_z,
                pt_pix_x, pt_pix_y,
                depth_val, diff);
            }

                // int3 idx3 = world2voxel(make_float3((float)pt_grid_x * param->vox_size, (float)pt_grid_y * param->vox_size, (float)pt_grid_z * param->vox_size), param->vox_size);
                // int3 idBlock = voxel2block(idx3);
                // printf("good pos\t(%f,%f,%f)\tblock\t(%d,%d,%d)\tvoxel\t(%d,%d,%d)\tcamera\t(%f,%f,%f)\tscreen\t(%d,%d)\tdepth\t%f\tdiff\t%f\n",
                // pt_base[0], pt_base[1], pt_base[2], 
                // idBlock.x, idBlock.y, idBlock.z,
                // idx3.x, idx3.y, idx3.z,
                // pt_cam_x, pt_cam_y, pt_cam_z,
                // pt_pix_x, pt_pix_y,
                // depth_val, diff);


            // Integrate
            int volume_idx = pt_grid_z * param->vox_dim.y * param->vox_dim.x + pt_grid_y * param->vox_dim.x + pt_grid_x;
            int image_idx = pt_pix_y * width + pt_pix_x;
            float dist = fmin(1.0f, diff / param->trunc_margin);
            float weight_old = weight[volume_idx];
            float weight_new = weight_old + 1.0f;
            weight[volume_idx] = weight_new;
            TSDF[volume_idx] = (TSDF[volume_idx] * weight_old + dist) / weight_new;
            TSDF_color[volume_idx * 3] = (TSDF_color[volume_idx * 3] * weight_old + rgb[3 * image_idx]) / weight_new;
            TSDF_color[volume_idx * 3 + 1] =
                    (TSDF_color[volume_idx * 3 + 1] * weight_old + rgb[3 * image_idx + 1]) / weight_new;
            TSDF_color[volume_idx * 3 + 2] =
                    (TSDF_color[volume_idx * 3 + 2] * weight_old + rgb[3 * image_idx + 2]) / weight_new;

            // printf("sdf\t%f\tcolor\t%d\t%d\t%d\tweight%f\n", TSDF[volume_idx],TSDF_color[volume_idx * 3],TSDF_color[volume_idx * 3 + 1],TSDF_color[volume_idx * 3 + 2],weight[volume_idx]);

        }
    }
  
    __host__ __device__
    bool operator==(const Vertex &a, const Vertex &b){
        return (a.x == b.x && a.y == b.y && a.z == b.z);
    }

    __device__ int d_floor(float f){
        if(f >= 0.0)
            return (int)f;
        else
            return (int)(f - 0.5);
    }

    __global__ 
    void marchingCubeHashKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
                unsigned int* d_valid_tri, unsigned int *d_heapBlockCounter, Triangle *tri, MarchingCubeParam *param){

        unsigned int idheap = blockIdx.x;

        if(idheap < *d_heapBlockCounter){
            int3 idBlock = dev_blockmap_.key_heap[idheap];
            VoxelBlock& vb = dev_blockmap_[idBlock];
            // Voxel& voxel = vb.voxels[threadIdx.x];
            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("vb  ==\n");

            int VOXEL_PER_BLOCK2 = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
            int z = threadIdx.x % VOXEL_PER_BLOCK + idBlock.z * VOXEL_PER_BLOCK;
            int y = ((threadIdx.x - z) % VOXEL_PER_BLOCK2) / VOXEL_PER_BLOCK + idBlock.y * VOXEL_PER_BLOCK;
            int x = threadIdx.x / VOXEL_PER_BLOCK2 + idBlock.x * VOXEL_PER_BLOCK;

            GRIDCELL grid;
            // printf("%d %d start \n",blockIdx.x, threadIdx.x);
            for (int k = 0; k < 8; ++k) {

                int cxi = x + param->idxMap[k][0];
                int cyi = y + param->idxMap[k][1];
                int czi = z + param->idxMap[k][2];

                int3 id_nb_block = voxel2block(make_int3(cxi,cyi,czi));

//                printf("current block%d %d %d  voxel %d %d %d  nbvoxel %d %d %d block %d %d %d\n", idBlock.x, idBlock.y, idBlock.z, x, y, z, cxi, cyi, czi, id_nb_block.x, id_nb_block.y, id_nb_block.z);
                if(cxi == x)
                    assert(idBlock.x == id_nb_block.x);
                if(cyi == y)
                    assert(idBlock.y == id_nb_block.y);                
                if(czi == z)
                    assert(idBlock.z == id_nb_block.z);

                int linear_id = voxelLinearInBlock(make_int3(cxi, cyi,czi), id_nb_block);

                if(dev_blockmap_.find(id_nb_block) != dev_blockmap_.end()){
                    grid.p[k] = Vertex(cxi, cyi, czi);

                    VoxelBlock& nb_block = dev_blockmap_[id_nb_block];

                    Voxel& nb_voxel = nb_block.voxels[linear_id];

                    grid.p[k].r = nb_voxel.sdf_color[0];
                    grid.p[k].g = nb_voxel.sdf_color[1];
                    grid.p[k].b = nb_voxel.sdf_color[2];
                    grid.val[k] = nb_voxel.sdf;

                } else {
//                    printf("相邻block不在\n");
//                     grid.p[k].r = 0;
//                     grid.p[k].g = 0;
//                     grid.p[k].b = 0;
//                     grid.val[k] = 0;
                     return;

                    // if(x == DEBUG_X && y == DEBUG_Y && z == DEBUG_Z)
                    // if(x % 4 == 0 && y % 4 == 0 && z % 4 == 0)
                    // if(blockIdx.x == 0 && threadIdx.x == 0)
                    //     printf("return here\n");
                    // printf("%d %d return \n",blockIdx.x, threadIdx.x);
//                    return;
                    // printf("%d\tn_____l\t%d\t%d\t%d\tsdf=%f\n", linear_id, cxi, cyi, czi, grid.val[k]);
                }
                // if(pt_grid_x == 20 && pt_grid_y == 8 && pt_grid_z == 0){
                //     printf("80 34 3  tsdf %f \n", grid.val[k]);
                // }
            }

            int cubeIndex = 0;
            // printf("%f %f %f %f %f %f %f %f\n",grid.val[0], grid.val[1],grid.val[2], grid.val[3], grid.val[4], grid.val[5],
            // grid.val[6], grid.val[7]);
            if (grid.val[0] < 0) cubeIndex |= 1;
            if (grid.val[1] < 0) cubeIndex |= 2;
            if (grid.val[2] < 0) cubeIndex |= 4;
            if (grid.val[3] < 0) cubeIndex |= 8;
            if (grid.val[4] < 0) cubeIndex |= 16;
            if (grid.val[5] < 0) cubeIndex |= 32;
            if (grid.val[6] < 0) cubeIndex |= 64;
            if (grid.val[7] < 0) cubeIndex |= 128;

            Vertex vertlist[12];
            if (param->edgeTable[cubeIndex] == 0)
                return;

            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("valid  ==\n");

            // if(x == 95 && y == 63 && z == 21)
            //     printf("\t%d\t%d\t%d\n", x, y, z);

            // printf("valid sdf\n");

            /* Find the vertices where the surface intersects the cube */
            if (param->edgeTable[cubeIndex] & 1)
                vertlist[0] =
                        VertexInterp(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (param->edgeTable[cubeIndex] & 2)
                vertlist[1] =
                        VertexInterp(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (param->edgeTable[cubeIndex] & 4)
                vertlist[2] =
                        VertexInterp(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (param->edgeTable[cubeIndex] & 8)
                vertlist[3] =
                        VertexInterp(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (param->edgeTable[cubeIndex] & 16)
                vertlist[4] =
                        VertexInterp(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 32)
                vertlist[5] =
                        VertexInterp(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 64)
                vertlist[6] =
                        VertexInterp(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (param->edgeTable[cubeIndex] & 128)
                vertlist[7] =
                        VertexInterp(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 256)
                vertlist[8] =
                        VertexInterp(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 512)
                vertlist[9] =
                        VertexInterp(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 1024)
                vertlist[10] =
                        VertexInterp(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 2048)
                vertlist[11] =
                        VertexInterp(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);


            int index = idheap * (VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK) + threadIdx.x;

            int count = 0;
            // int addr = 0;
            for (int ti = 0; param->triTable[cubeIndex][ti] != -1; ti += 3) {
                // addr = atomicAdd(&d_valid_tri[0], 1);
                // tri[addr].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                // tri[addr].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                // tri[addr].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                // tri[addr].valid = true;
                tri[index * 5 + count].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                tri[index * 5 + count].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                tri[index * 5 + count].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                tri[index * 5 + count].valid = true;

                // 会存在无法构成三角形的情况吗。
                if(tri[index * 5 + count].p[0] == tri[index * 5 + count].p[1] ||
                    tri[index * 5 + count].p[1] == tri[index * 5 + count].p[2] ||
                    tri[index * 5 + count].p[0] == tri[index * 5 + count].p[2])
                    tri[index * 5 + count].valid = false;

                count++;
            }
        }
    }

    __global__
    void marchingCubeKernel(float *TSDF, unsigned char *TSDF_color, Triangle *tri, MarchingCubeParam *param) {
        int pt_grid_z = blockIdx.x;
        int pt_grid_y = threadIdx.x;

        int global_index = pt_grid_z * param->vox_dim.x * param->vox_dim.y - pt_grid_y * param->vox_dim.x;

        for (int pt_grid_x = 0; pt_grid_x < param->vox_dim.x; ++pt_grid_x) {
            int index = global_index + pt_grid_x;

            GRIDCELL grid;
            for (int k = 0; k < 8; ++k) {
                int cxi = pt_grid_x + param->idxMap[k][0];
                int cyi = pt_grid_y + param->idxMap[k][1];
                int czi = pt_grid_z + param->idxMap[k][2];
                grid.p[k] = Vertex(cxi, cyi, czi);
                grid.p[k].r = TSDF_color[3 * (czi * param->vox_dim.y * param->vox_dim.z +
                                              cyi * param->vox_dim.z + cxi)];
                grid.p[k].g = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 1];
                grid.p[k].b = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 2];
                grid.val[k] = TSDF[czi * param->vox_dim.y * param->vox_dim.z +
                                   cyi * param->vox_dim.z +
                                   cxi];

                if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                    printf("nbvoxel\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);

            }

            int cubeIndex = 0;
            if (grid.val[0] < 0) cubeIndex |= 1;
            if (grid.val[1] < 0) cubeIndex |= 2;
            if (grid.val[2] < 0) cubeIndex |= 4;
            if (grid.val[3] < 0) cubeIndex |= 8;
            if (grid.val[4] < 0) cubeIndex |= 16;
            if (grid.val[5] < 0) cubeIndex |= 32;
            if (grid.val[6] < 0) cubeIndex |= 64;
            if (grid.val[7] < 0) cubeIndex |= 128;

            Vertex vertlist[12];
            if (param->edgeTable[cubeIndex] == 0)
                continue;

            /* Find the vertices where the surface intersects the cube */
            if (param->edgeTable[cubeIndex] & 1)
                vertlist[0] =
                        VertexInterp(0, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
            if (param->edgeTable[cubeIndex] & 2)
                vertlist[1] =
                        VertexInterp(0, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
            if (param->edgeTable[cubeIndex] & 4)
                vertlist[2] =
                        VertexInterp(0, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
            if (param->edgeTable[cubeIndex] & 8)
                vertlist[3] =
                        VertexInterp(0, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
            if (param->edgeTable[cubeIndex] & 16)
                vertlist[4] =
                        VertexInterp(0, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 32)
                vertlist[5] =
                        VertexInterp(0, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 64)
                vertlist[6] =
                        VertexInterp(0, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
            if (param->edgeTable[cubeIndex] & 128)
                vertlist[7] =
                        VertexInterp(0, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 256)
                vertlist[8] =
                        VertexInterp(0, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
            if (param->edgeTable[cubeIndex] & 512)
                vertlist[9] =
                        VertexInterp(0, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
            if (param->edgeTable[cubeIndex] & 1024)
                vertlist[10] =
                        VertexInterp(0, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
            if (param->edgeTable[cubeIndex] & 2048)
                vertlist[11] =
                        VertexInterp(0, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

            int count = 0;
            for (int ti = 0; param->triTable[cubeIndex][ti] != -1; ti += 3) {
                tri[index * 5 + count].p[0] = vertlist[param->triTable[cubeIndex][ti]];
                tri[index * 5 + count].p[1] = vertlist[param->triTable[cubeIndex][ti + 1]];
                tri[index * 5 + count].p[2] = vertlist[param->triTable[cubeIndex][ti + 2]];
                tri[index * 5 + count].valid = true;
                count++;
            }

            // if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                // printf("mesh\t%d\t%d\t%d\tcount=%d\n", pt_grid_x, pt_grid_y, pt_grid_z, count);

            if(pt_grid_x % 4 == 0 && pt_grid_y % 4 == 0 && pt_grid_z % 4 == 0){

            for (int k = 0; k < 8; ++k) {
                int cxi = pt_grid_x + param->idxMap[k][0];
                int cyi = pt_grid_y + param->idxMap[k][1];
                int czi = pt_grid_z + param->idxMap[k][2];
                grid.p[k] = Vertex(cxi, cyi, czi);
                grid.p[k].r = TSDF_color[3 * (czi * param->vox_dim.y * param->vox_dim.z +
                                              cyi * param->vox_dim.z + cxi)];
                grid.p[k].g = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 1];
                grid.p[k].b = TSDF_color[
                        3 * (czi * param->vox_dim.y * param->vox_dim.z + cyi * param->vox_dim.z + cxi) + 2];
                grid.val[k] = TSDF[czi * param->vox_dim.y * param->vox_dim.z +
                                   cyi * param->vox_dim.z +
                                   cxi];

                // if(pt_grid_x == DEBUG_X && pt_grid_y == DEBUG_Y && pt_grid_z == DEBUG_Z)
                    // printf("mesh\t%d\t%d\t%d\tr%dg%db%d\tsdf%f\n", cxi, cyi, czi,grid.p[k].r,grid.p[k].g,grid.p[k].b, grid.val[k]);

            }
        }

            assert(count != 0);
        }
    }

    __host__
    void GpuTsdfGenerator::clearheap(){
        // unsigned int src = 0;
        // // cudaSafeCall(cudaFree(d_heapBlockCounter));
        // cudaSafeCall(cudaMemcpy(d_heapBlockCounter, &src, sizeof(unsigned int), cudaMemcpyHostToDevice));
        // std::cout<<" d_heapBlockCounter clear "<<std::endl;
        dev_blockmap_->clearheap();
        dev_blockmap_chunks->clearheap();
        // d_heapBlockCounter = 0;
    }

    __global__ void RidOfPlane(float* depth, float* planeParam) {

        int id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id >= 640 * 480)
            return;

        float fx = 0.00193256, fy = 0.00193256, cx = -0.59026608, cy = -0.48393462;
        float inv[] = {fx, cx, fy, cy};

        float row = id / 640 + 0.5f;
        float col = id % 640 + 0.5f;

        float pointCamera[]  =  {depth[id] * (inv[0] * col + inv[1]),
                            depth[id] * (inv[2] * row + inv[3]),
                            depth[id]};

        if(fabs(pointCamera[0] * planeParam[0] + pointCamera[1] * planeParam[1] +
             pointCamera[2] * planeParam[2]  + planeParam[3]) < PLANE_TRUNCATION_VALUE)
            depth[id] = 0.f;
    }

    __host__
    GpuTsdfGenerator::GpuTsdfGenerator(int width, int height, float fx, float fy, float cx, float cy, float max_depth,
                                       float origin_x = -1.5f, float origin_y = -1.5f, float origin_z = 0.5f,
                                       float vox_size = 0.006f, float trunc_m = 0.03f, int vox_dim_x = 500,
                                       int vox_dim_y = 500, int vox_dim_z = 500) {
                                           //vox_size 表征每个格子所对应的现实场景的长度

        std::cout<<" GpuTsdfGenerator init "<<std::endl;

        checkCUDA(__LINE__, cudaGetLastError());

        im_width_ = width;
        im_height_ = height;

        memset(K_, 0.0f, sizeof(float) * 3 * 3);
        //K_ 不是应该为k的逆吗 这个地方需要后续留意
        K_[0] = fx;
        K_[2] = cx;
        K_[4] = fy;
        K_[5] = cy;
        K_[8] = 1.0f;

        std::cout<<" hashing init "<<std::endl;

        /***                     ----- hashing parameters -----             ***/
        chunk_size = BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * vox_size;//chunk的实际长度

        // cudaSafeCall(cudaMalloc(&d_outBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        // std::cout<<" d_outBlock init "<<std::endl;
        // cudaSafeCall(cudaMalloc(&d_outBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        // std::cout<<" d_outBlockPos init "<<std::endl;
        // MAX_CPU2GPU_BLOCKS 这个参数是否必须要现在设置
        cudaSafeCall(cudaMalloc(&d_inBlock, sizeof(VoxelBlock) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlock init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPos, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPos init "<<std::endl;
        cudaSafeCall(cudaMalloc(&d_inBlockPosHeap, sizeof(VoxelBlockPos) * MAX_CPU2GPU_BLOCKS));
        std::cout<<" d_inBlockPosHeap init "<<std::endl;
        // cudaSafeCall(cudaMalloc(&d_outBlockCounter, sizeof(unsigned int)));
        // std::cout<<" d_outBlockCounter init "<<std::endl;
        d_heapBlockCounter = NULL;
        // heapCounter initial;
        h_heapBlockCounter = 0;
        cudaSafeCall(cudaMalloc((void**)&d_heapBlockCounter, sizeof(unsigned int)));

        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>
        //     blocks(10000, 2, 19997, int3{999999, 999999, 999999});

        // dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_ init "<<std::endl;

        // std::cout<<" d_heapBlockCounter init "<<std::endl;
        // dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(100001, 4, 400000, int3{999999, 999999, 999999});
        // std::cout<<" dev_blockmap_chunks init "<<std::endl;

        h_chunks = new Chunk[MAX_CHUNK_NUM * MAX_CHUNK_NUM * MAX_CHUNK_NUM];        
        std::cout<<" h_chunks init "<<std::endl;
        // /***                            ----- end  -----                    ***/

        // for(int i = 0; i < 9; i ++)
        //     std::cout<<K_[i]<<" - "<<std::endl;
        // std::cout<<std::endl;

        param_ = new MarchingCubeParam();

        param_->vox_origin.x = origin_x;
        param_->vox_origin.y = origin_y;
        param_->vox_origin.z = origin_z;

        param_->vox_dim.x = vox_dim_x;
        param_->vox_dim.y = vox_dim_y;
        param_->vox_dim.z = vox_dim_z;

        param_->max_depth = max_depth;
        //单位是mm
        param_->min_depth = 0.1;

        param_->vox_size = vox_size;

        param_->trunc_margin = trunc_m;

        // 该值范围为要重建物体的实际大小(voxel_size * num > 需要实际重建的实际大小！！！！)
        param_->total_vox = param_->vox_dim.x * param_->vox_dim.y * param_->vox_dim.z;

        param_->block_size = VOXEL_PER_BLOCK * vox_size;

        param_->im_width = width;
        param_->im_height = height;
        param_->fx = fx;
        param_->fy = fy;
        param_->cx = cx;
        param_->cy = cy;

        // // Initialize voxel grid
         TSDF_ = new float[param_->total_vox];
        // TSDF_color_ = new unsigned char[3 * param_->total_vox];
         weight_ = new float[param_->total_vox];
         memset(TSDF_, .0f, sizeof(float) * param_->total_vox);
//         memset(TSDF_color_, 0.0f, 3 * sizeof(unsigned char) * param_->total_vox);
         memset(weight_, 0.0f, sizeof(float) * param_->total_vox);

        tri_ = (Triangle *) malloc(sizeof(Triangle) * param_->total_vox * 5);
        
        // Load variables to GPU memory
        cudaMalloc(&dev_param_, sizeof(MarchingCubeParam));
        cudaMalloc(&dev_TSDF_, param_->total_vox * sizeof(float));
//        cudaMalloc(&dev_TSDF_color_, 3 * param_->total_vox * sizeof(unsigned char));
        cudaMalloc(&dev_weight_, param_->total_vox * sizeof(float));
        // cudaMalloc(&dev_tri_, sizeof(Triangle) * param_->total_vox * 5);
        checkCUDA(__LINE__, cudaGetLastError());
        //该行外放
//        cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_TSDF_, TSDF_,
                   // param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_TSDF_color_, TSDF_color_,
        //            3 * param_->total_vox * sizeof(unsigned char),
        //            cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_weight_, weight_,
        //            param_->total_vox * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDA(__LINE__, cudaGetLastError());

        cudaMalloc(&dev_K_, 3 * 3 * sizeof(float));
        cudaMemcpy(dev_K_, K_, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_c2w_, 4 * 4 * sizeof(float));
        cudaMalloc(&dev_w2c_, 4 * 4 * sizeof(float));
        cudaMalloc(&dev_depth_, im_height_ * im_width_ * sizeof(float));
        cudaMalloc(&dev_rgb_, 3 * im_height_ * im_width_ * sizeof(unsigned char));
        checkCUDA(__LINE__, cudaGetLastError());
    }

    // __host__
    // void inverse_matrix(float* m, float* minv){
    //     // computes the inverse of a matrix m
    //     std::cout<<" m = "<<std::endl;

    //     for(int r=0;r<3;++r){
    //         for(int c=0;c<3;++c){
    //             // m[r * 3 + c] = c2w_[r * 4 + c];
    //             printf("%f\t",m[r * 3 + c]);
    //         }
    //         std::cout<<std::endl;
    //     }

    //     double det = m[0 * 3 + 0] * (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) -
    //                  m[0 * 3 + 1] * (m[1 * 3 + 0] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 0]) +
    //                  m[0 * 3 + 2] * (m[1 * 3 + 0] * m[2 * 3 + 1] - m[1 * 3 + 1] * m[2 * 3 + 0]);

    //     double invdet = 1 / det;

    //     std::cout<<" inverse "<<std::endl;

    //     // inverse_matrix(m, minv);
    //     for(int r=0;r<3;++r){
    //         for(int c=0;c<3;++c){
    //             printf("%f\t",minv[r * 3 + c]);
    //         }
    //         std::cout<<std::endl;
    //     }

    //     minv[0 * 3 + 0] = (m[1 * 3 + 1] * m[2 * 3 + 2] - m[2 * 3 + 1] * m[1 * 3 + 2]) * invdet;
    //     minv[0 * 3 + 1] = (m[0 * 3 + 2] * m[2 * 3 + 1] - m[0 * 3 + 1] * m[2 * 3 + 2]) * invdet;
    //     minv[0 * 3 + 2] = (m[0 * 3 + 1] * m[1 * 3 + 2] - m[0 * 3 + 2] * m[1 * 3 + 1]) * invdet;
    //     minv[1 * 3 + 0] = (m[1 * 3 + 2] * m[2 * 3 + 0] - m[1 * 3 + 0] * m[2 * 3 + 2]) * invdet;
    //     minv[1 * 3 + 1] = (m[0 * 3 + 0] * m[2 * 3 + 2] - m[0 * 3 + 2] * m[2 * 3 + 0]) * invdet;
    //     minv[1 * 3 + 2] = (m[1 * 3 + 0] * m[0 * 3 + 2] - m[0 * 3 + 0] * m[1 * 3 + 2]) * invdet;
    //     minv[2 * 3 + 0] = (m[1 * 3 + 0] * m[2 * 3 + 1] - m[2 * 3 + 0] * m[1 * 3 + 1]) * invdet;
    //     minv[2 * 3 + 1] = (m[2 * 3 + 0] * m[0 * 3 + 1] - m[0 * 3 + 0] * m[2 * 3 + 1]) * invdet;
    //     minv[2 * 3 + 2] = (m[0 * 3 + 0] * m[1 * 3 + 1] - m[1 * 3 + 0] * m[0 * 3 + 1]) * invdet;
    // }


    __host__
    void GpuTsdfGenerator::getLocalGrid(){
    }

    __host__
    void GpuTsdfGenerator::processFrame(float *depth, unsigned char *rgb, float *c2w,
            const Eigen::Vector4f& planeParam) {

        // 造成问题的原因：读取格式为double，有时被误设为float
//        for (int x = 0; x<16; x++)
//            std::cout << c2w[x] << " ";
//        std::cout << std::endl;

        printf("planeparam is %f %f %f %f\n", planeParam[0], planeParam[1], planeParam[2], planeParam[3]);

        system_clock::time_point startTimePerFrame = system_clock::now();
//        dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(2001, 10, 10000, int3{999999, 999999, 999999});

        dev_blockmap_ = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(2001, 10, 5000, int3{999999, 999999, 999999});
        std::cout<<" dev_blockmap_ init "<<std::endl;

        cudaDeviceSynchronize();
        std::cout<<" d_heapBlockCounter init "<<std::endl;
//        dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(3001, 10,30000, int3{999999, 999999, 999999});
        dev_blockmap_chunks = new vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::device_memspace>(2001, 10,12000, int3{999999, 999999, 999999});
        std::cout<<" dev_blockmap_chunks init "<<std::endl;
        cudaDeviceSynchronize();

        float temPlane[] = {planeParam[0], planeParam[1], planeParam[2], planeParam[3]};
        std::cout << 1 <<std::endl;

        auto middleT = system_clock::now();

        ark::printTime(startTimePerFrame, middleT, "voxelBlock");
        
        // std::cout<<"k_"<<K_[0]<<K_[4]<<K_[7]<<std::endl;
        // std::cout<<im_height_<<" "<<im_width_<<std::endl;
        // std::cout<<depth[0]<<" "<<depth[10]<<" "<<depth[100]<<std::endl;
        //dev_param_ 参数设置。
        cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);
        
        cudaMemcpy(dev_c2w_, c2w, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
        float* ridOfPlane;
        cudaMalloc(&ridOfPlane, 4 * sizeof(float));
        cudaMemcpy(ridOfPlane, temPlane, 4* sizeof(float), cudaMemcpyHostToDevice);


        //是否成功复制
        printf("The error after hash Memcpy%s\n", cudaGetErrorName(cudaGetLastError()));
        std::cout<<2<<std::endl;
        cudaDeviceSynchronize();
        cudaMemcpy(dev_depth_, depth, im_height_ * im_width_ * sizeof(float), cudaMemcpyHostToDevice);

        auto ridPlaneTime = system_clock::now();
        const dim3 gridDepth(300,1);
        const dim3 blockDepth(1024,1);

        RidOfPlane<<<gridDepth, blockDepth>>>(dev_depth_, ridOfPlane);
        cudaDeviceSynchronize();

        auto endRidPlaneTime = system_clock::now();

        //清除地板背景耗时很小
//        ark::printTime(ridPlaneTime, endRidPlaneTime, "清除地板背景所用时间");

        std::cout << 3 << std::endl;
        printf("The error before streaminCPU is %s\n", cudaGetErrorName(cudaGetLastError()));
        cudaDeviceSynchronize();
        std::cout << 2 << std::endl;

        cudaMemcpy(dev_rgb_, rgb, 3 * im_height_ * im_width_ * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        // checkCUDA(__LINE__, cudaGetLastError());
        memcpy(c2w_, c2w, 4 * 4 * sizeof(float));
        std::cout<< "clear h_heapBlockCounter" <<std::endl;
        auto a = system_clock::now();
        ark::printTime(middleT, a, "分配其余显内存所用时间\n");
        system_clock::time_point b;
        a = system_clock::now();
        clearheap();
        b = system_clock::now();
        ark::printTime(a, b, "clearHeap所用时间");

        a = system_clock::now();
        //验证第一步， streaminGPU部分
        streamInCPU2GPU(dev_K_, dev_c2w_, dev_depth_);
        b = system_clock::now();

        ark::printTime(a, b, "从CPU流入GPU所用时间");
        //验证第一步， streaminGPU部分
        cudaDeviceSynchronize();
        // checkCUDA(__LINE__, cudaGetLastError());
        printf("The error after streaminCPU is %s\n", cudaGetErrorName(cudaGetLastError()));
        std::cout<< "cpu2gpu 成功" <<std::endl;
        std::cout<<"h_heapBlockCounter = "<<h_heapBlockCounter<<std::endl;
        a = system_clock::now();
        HashAssign(dev_depth_, im_height_, im_width_, dev_param_, dev_K_, dev_c2w_);
        cudaDeviceSynchronize();
        b = system_clock::now();

        ark::printTime(a, b, "hashassign 安排所用时间");
        // HashAlloc();
        // checkCUDA(__LINE__, cudaGetLastError());
        a = system_clock::now();
        printf("The error after hashassigned%s\n", cudaGetErrorName(cudaGetLastError()));
        std::cout<< "HashAssign 成功" <<std::endl;
        Triangle *dev_hash_tri_ = nullptr;
        unsigned int *d_valid_tri; //设备端三角形数目
        unsigned int h_valid_tri = 0;
        cudaSafeCall(cudaMalloc(&d_valid_tri, sizeof(unsigned int)));
        int tri_0;
        cudaSafeCall(cudaMemcpy(d_valid_tri, &tri_0, sizeof(unsigned int), cudaMemcpyHostToDevice));

        cudaSafeCall(cudaMemcpy(&h_heapBlockCounter,d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::cout<<"h_heapBlockCounter = "<<h_heapBlockCounter<<std::endl;

        unsigned int total_vox = h_heapBlockCounter * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;

        {
            //没看出这个mutex 是锁什么的。
//            std::unique_lock<std::mutex> lock(tsdf_mutex_);
            // printf("The heapblockcounter is %d\n", h_heapBlockCounter);
            // printf("h_heapBlockCounter is %d\n", h_heapBlockCounter);
            cudaDeviceSynchronize();
            const dim3 gridSize(h_heapBlockCounter, 1);
//            const dim3 blockSize(256, 1);
            const dim3 blockSize(VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK, 1);
            // checkCUDA(__LINE__, cudaGetLastError());

            // std::cout<<"start IntegrateHashKernel"<<std::endl;
            IntegrateHashKernel <<< gridSize, blockSize >>> (dev_K_, dev_c2w_, dev_depth_, dev_rgb_,
                im_height_, im_width_, dev_param_, *dev_blockmap_, d_inBlockPosHeap, d_heapBlockCounter,
                    ridOfPlane);

            // checkCUDA(__LINE__, cudaGetLastError());
            // std::cout<<"finish IntegrateHashKernel"<<std::endl;

            cudaDeviceSynchronize();
            //进行异常block收集
//            DropGarbage( h_heapBlockCounter, d_heapBlockCounter);
//            cudaDeviceSynchronize();


            b = system_clock::now();
            ark::printTime(a, b, "tsdf 所用时间");

            a = system_clock::now();
            cudaMalloc(&dev_hash_tri_, sizeof(Triangle) * total_vox * 5);
            cudaMemset(dev_hash_tri_, 0, sizeof(Triangle) * total_vox * 5);

            // std::cout<<"start marchingCubeHashKernel"<<std::endl;
        
            marchingCubeHashKernel <<< gridSize, blockSize >>>
                                                       (*dev_blockmap_, d_valid_tri, d_heapBlockCounter, dev_hash_tri_, dev_param_);
            // checkCUDA(__LINE__, cudaGetLastError());

            cudaSafeCall(cudaMemcpy(&h_valid_tri, d_valid_tri, sizeof(unsigned int), cudaMemcpyDeviceToHost));

            // std::cout<<"finish marchingCubeHashKernel"<<std::endl;
        }
        std::cout<< "tsdf 成功" <<std::endl;

        cudaDeviceSynchronize();

        hash_tri_ = (Triangle *) malloc(sizeof(Triangle) * total_vox * 5);


        // cudaSafeCall(cudaMemcpy(h_inBlockPosHeap, dev_blockmap_->key_heap, sizeof(VoxelBlockPos) * h_heapBlockCounter, cudaMemcpyDeviceToHost));

        {
            std::unique_lock<std::mutex> lock(tri_mutex_);
            std::unique_lock<std::mutex> lockchunk(chunk_mutex_);
            // cudaMemcpy(tri_, dev_hash_tri_, sizeof(Triangle) * total_vox * 5, cudaMemcpyDeviceToHost);
            // checkCUDA(__LINE__, cudaGetLastError());
            cudaMemcpy(hash_tri_, dev_hash_tri_, sizeof(Triangle) * total_vox * 5, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            // checkCUDA(__LINE__, cudaGetLastError());
//            showPerFrame(hash_tri_, total_vox*5);
            tri2mesh( "model" + std::to_string(plyCnt++) + ".ply", hash_tri_, total_vox);
        }





        b = system_clock::now();
        ark::printTime(a, b, "marching cubes时间");
        a = system_clock::now();
        streamOutGPU2CPU();
        b = system_clock::now();
        ark::printTime(a, b, "streamoutgpu");
        printf("end streamOut\n");

//        cudaFree(dev_hash_tri_);
//        cudaFree(d_valid_tri);
//        delete dev_blockmap_;
//        delete dev_blockmap_chunks;

        printf("%s\n", cudaGetErrorName(cudaGetLastError()));
        countFrame ++;
        printf("count++\n");
        system_clock::time_point endTimePerFrame = system_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(endTimePerFrame - startTimePerFrame).count();
        printf("每一帧的时间消耗为 %f s\n", (float)duration * microseconds::period::num / microseconds::period::den);
         if(h_heapBlockCounter){
            delete dev_blockmap_;
            delete dev_blockmap_chunks;
            cudaFree(dev_hash_tri_);
            cudaFree(d_valid_tri);
        }
    }

    __host__ 
    void GpuTsdfGenerator::insert_tri(){

    }

    __host__
    void GpuTsdfGenerator::Shutdown() {
        free(tri_);
        // cudaFree(dev_TSDF_);
        // cudaFree(dev_TSDF_color_);
        // cudaFree(dev_weight_);
        cudaFree(dev_K_);
        cudaFree(dev_c2w_);
        cudaFree(dev_depth_);
        cudaFree(dev_rgb_);
        // cudaFree(dev_tri_);
        cudaFree(dev_param_);
        cudaFree(d_inBlock);
        cudaFree(d_inBlockPos);
        cudaFree(d_inBlockPosHeap);

        delete dev_blockmap_;
        delete dev_blockmap_chunks;
    }
    //插值函数 同时也会对表面上色（插值tsdf值的同时为点附上了颜色)

    __host__ __device__
    Vertex VertexInterp(const float& isolevel, const Vertex& p1, const Vertex& p2, const float& valp1, const float& valp2) {
        float mu;
        Vertex p;

        if (fabs(isolevel - valp1) < 0.00001)
            return p1;
        if (fabs(isolevel - valp2) < 0.00001)
            return p2;
        if (fabs(valp1 - valp2) < 0.00001)
            return p1;
        mu = (isolevel - valp1) / (valp2 - valp1);
//        printf("valp1 is %f valp2 is %f\n", valp1, valp2);
        p.x = p1.x + mu * (p2.x - p1.x);
        p.y = p1.y + mu * (p2.y - p1.y);
        p.z = p1.z + mu * (p2.z - p1.z);
        p.r = p1.r + mu * (p2.r - p1.r);
        p.g = p1.g + mu * (p2.g - p1.g);
        p.b = p1.b + mu * (p2.b - p1.b);

        return p;
    }

    __host__
    void GpuTsdfGenerator::SaveTSDF(std::string filename) {
        std::unique_lock<std::mutex> lock(tsdf_mutex_);
        // Load TSDF voxel grid from GPU to CPU memory
        cudaMemcpy(TSDF_, dev_TSDF_,
                   param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(TSDF_color_, dev_TSDF_color_,
                   3 * param_->total_vox * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(weight_, dev_weight_,
                   param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDA(__LINE__, cudaGetLastError());
        // Save TSDF voxel grid and its parameters to disk as binary file (float array)
        std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
        std::string voxel_grid_saveto_path = filename;
        std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
        float vox_dim_xf = (float) param_->vox_dim.x;
        float vox_dim_yf = (float) param_->vox_dim.y;
        float vox_dim_zf = (float) param_->vox_dim.z;
        outFile.write((char *) &vox_dim_xf, sizeof(float));
        outFile.write((char *) &vox_dim_yf, sizeof(float));
        outFile.write((char *) &vox_dim_zf, sizeof(float));
        outFile.write((char *) &param_->vox_origin.x, sizeof(float));
        outFile.write((char *) &param_->vox_origin.y, sizeof(float));
        outFile.write((char *) &param_->vox_origin.z, sizeof(float));
        outFile.write((char *) &param_->vox_size, sizeof(float));
        outFile.write((char *) &param_->trunc_margin, sizeof(float));
        for (int i = 0; i < param_->total_vox; ++i)
            outFile.write((char *) &TSDF_[i], sizeof(float));
        for (int i = 0; i < 3 * param_->total_vox; ++i)
            outFile.write((char *) &TSDF_color_[i], sizeof(unsigned char));
        outFile.close();
    }

    __host__
    void GpuTsdfGenerator::SavePLY(std::string filename, const cv::Rect& foreground) {
        // {
        //     std::unique_lock<std::mutex> lock(tsdf_mutex_);
        //     cudaMemcpy(TSDF_, dev_TSDF_,
        //                param_->total_vox * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(TSDF_color_, dev_TSDF_color_,
        //                3 * param_->total_vox * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        //     checkCUDA(__LINE__, cudaGetLastError());
        // }
        tsdf2mesh(filename, foreground);
    }

    __host__
    void showPerFrame(Triangle * tri_, int triangleNum) {

        for(int i = 0; i < triangleNum; i++) {

            if(!tri_[i].valid)
                continue;
            printf("%d %f %f %f\n", i, tri_[i].p[0].x, tri_[i].p[0].y, tri_[i].p[0].z);
            // std::cout<<"render"<<std::endl;
//            glBegin(GL_TRIANGLES);
//            for (int j = 0; j < 3; ++j) {
//                glColor3f(tri_[i].p[j].r / 255.f, tri_[i].p[j].g / 255.f, tri_[i].p[j].b / 255.f);
////                                glVertex3f(10 * tri_[i].p[j].x * param_->vox_size - 25,
////                                           10 * tri_[i].p[j].y * param_->vox_size - 25,
////                                           10 * tri_[i].p[j].z * param_->vox_size - 20);
//                glVertex3f(10 * tri_[i].p[j].x * 2 * 0.01,
//                           -10 * tri_[i].p[j].y * 2 * 0.01,
//                           10 * tri_[i].p[j].z * 2 * 0.01);
//            }
//            glEnd();
        }
        return;
    }

    __host__
    void GpuTsdfGenerator::render() {
//        return;

        int chunk_half = MAX_CHUNK_NUM / 2;
        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        int tri_num = total_vox * 5;
        // int empty = 0;
        for(int x = - chunk_half; x < chunk_half; x ++){
            for(int y = - chunk_half; y < chunk_half; y ++){
                for(int z = - chunk_half; z < chunk_half; z ++){

                    std::unique_lock<std::mutex> lock(chunk_mutex_);
                    int id = chunkGetLinearIdx(x,y,z);
                    Triangle* tri_ = h_chunks[id].tri_;
                    // printf("what is %d\n", tri_[0].p[0].x);
                    if(tri_ != nullptr){
                        
                        int count = 0;
                        for(int i = 0; i < tri_num; i ++){
                            // std::cout<<h_chunks[id].tri_[i].valid<<std::endl;
                            if(!h_chunks[id].tri_[i].valid)
                                continue;
                            count ++;
                            // std::cout<<"render"<<std::endl;
                            glBegin(GL_TRIANGLES);
                            for (int j = 0; j < 3; ++j) {
                                glColor3f(tri_[i].p[j].r / 255.f, tri_[i].p[j].g / 255.f, tri_[i].p[j].b / 255.f);
//                                glVertex3f(10 * tri_[i].p[j].x * param_->vox_size - 25,
//                                           10 * tri_[i].p[j].y * param_->vox_size - 25,
//                                           10 * tri_[i].p[j].z * param_->vox_size - 20);
                                glVertex3f(10 * tri_[i].p[j].x * param_->vox_size * 0.01,
                                           -10 * tri_[i].p[j].y * param_->vox_size * 0.01,
                                           10 * tri_[i].p[j].z * param_->vox_size * 0.01);
                            }
                            glEnd();
                        }
                        if(count == 0) {
//                            if (h_chunks[id].isOccupied == true)
//                                printf("chunk.isOccupied is false\n");
                            h_chunks[id].isOccupied = false;
                        }

                    }

                }
            }
        }
    }

    __host__
    void tri2mesh(std::string outputFileName, Triangle* tri_, int triNum) {

        std::vector<Face> faces;
        std::vector<Vertex> vertices;

        int validCount = 0;

        std::unordered_map<Vertex, int, VertexHasher, VertexEqual> vertexHashTable;

        for(int k = 0; k < triNum; k++){
            int flag = 0;
            for(int i = 0; i < 5; i ++){
                int pi = 5 * k + i;
                if(!tri_[pi].valid)
                    continue;

                flag = 1;
                Face f;
                for (int j = 0; j < 3; ++j) {

                    if(vertexHashTable.find(tri_[pi].p[j]) == vertexHashTable.end()){
                        unsigned int count = vertices.size();
                        vertexHashTable[tri_[pi].p[j]] = count;
                        f.vIdx[j] = count;
                        Vertex vp = tri_[pi].p[j];
                        vp.x *= 2 ;
                        vp.y *= 2 ;
                        vp.z *= 2 ;

//                                        std::cout << vp.x << " " <<vp.y << std::endl;
                        vertices.push_back(vp);
                    } else{
                        f.vIdx[j] = vertexHashTable[tri_[pi].p[j]];
                    }
                }
                if(flag)
                    faces.push_back(f);
            }
            if(flag)
                validCount ++;
        }


        std::cout << vertices.size() << std::endl;
        std::ofstream plyFile;
        plyFile.open(outputFileName);
        plyFile << "ply\nformat ascii 1.0\ncomment stanford bunny\nelement vertex ";
        plyFile << vertices.size() << "\n";
        plyFile << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
        plyFile << "element face " << faces.size() << "\n";
        plyFile << "property list uchar int vertex_index\nend_header\n";
        for (auto v : vertices) {
            plyFile << v.x << " " << v.y << " " << v.z << " " << (int) v.r << " " << (int) v.g << " " << (int) v.b
                    << "\n";

//            plyFile << v.x << " " << v.y << " " << v.z << "\n";
        }
        for (auto f : faces) {
            plyFile << "3 " << f.vIdx[0] << " " << f.vIdx[1] << " " << f.vIdx[2] << "\n";
        }
        plyFile.close();
        std::cout << "File saved" << std::endl;
//        std::cout << "totalsize = "<< totalsize << " valid = "<< validCount << std::endl;
    }

    __host__
    void GpuTsdfGenerator::tsdf2mesh(std::string outputFileName, const cv::Rect& foreground) {
//        printf("%f %f %f %f\n", foreground.x, foreground.y, foreground.width, foreground.height);

//        float minX = foreground.x * 1.0 / param_->vox_size,
//        maxX = (foreground.x + foreground.width) * 1.0 / param_->vox_size,
//        minY = foreground.y * 1.0 / param_->vox_size,
//        maxY = (foreground.y + foreground.height) * 1.0 / param_->vox_size;

        std::vector<Face> faces;
        std::vector<Vertex> vertices;

        // std::unordered_map<std::string, int> verticesIdx;
        // std::vector<std::list<std::pair<Vertex, int>>> hash_table(param_->total_vox,
                                                                  // std::list<std::pair<Vertex, int>>());
        std::unique_lock<std::mutex> lock(chunk_mutex_);

        // int vertexCount = 0;
        // int emptyCount = 0;
        int totalsize = 0;
        int validCount = 0;

        std::cout << "Start saving ply, totalsize: " << param_->total_vox << std::endl;

        std::unordered_map<Vertex, int, VertexHasher, VertexEqual> vertexHashTable;

        int chunk_half = MAX_CHUNK_NUM / 2;
        int block_total = BLOCK_PER_CHUNK * BLOCK_PER_CHUNK * BLOCK_PER_CHUNK;
        int total_vox = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * block_total;
        // int tri_num = total_vox * 5;

        int VoxelPerChunk = BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * BLOCK_PER_CHUNK * VOXEL_PER_BLOCK * BLOCK_PER_CHUNK * VOXEL_PER_BLOCK;

        for(int x = - chunk_half; x < chunk_half; x ++){
            for(int y = - chunk_half; y < chunk_half; y ++){
                for(int z = - chunk_half; z < chunk_half; z ++){
                    int id = chunkGetLinearIdx(x,y,z);
                    // 暂时注释
//                    if(h_chunks[id].isOccupied == false)
//                        h_chunks[id].release();

                    // printf("id chunk %d \n", id);
                    Triangle* tri_ = h_chunks[id].tri_;
                    if(tri_ != nullptr){
                        totalsize += VoxelPerChunk;


                        // int count = 0;

                        for(int k = 0; k < total_vox; k++){
                            int flag = 0;
                            for(int i = 0; i < 5; i ++){
                                int pi = 5 * k + i;
                                if(!h_chunks[id].tri_[pi].valid)
                                    continue;

                                flag = 1;
                                Face f;
                                for (int j = 0; j < 3; ++j) {
                                    //根据手动画框方法提前前景
//                                    if(!(tri_[pi].p[j].x >= minY && tri_[pi].p[j].x <= maxY &&
//                                            tri_[pi].p[j].y >= minX && tri_[pi].p[j].y <= maxX)) {
//                                        flag = 0;
//                                        break;
//                                    }

                                    if(vertexHashTable.find(tri_[pi].p[j]) == vertexHashTable.end()){
                                        unsigned int count = vertices.size();
                                        vertexHashTable[tri_[pi].p[j]] = count;
                                        f.vIdx[j] = count;
                                        Vertex vp = tri_[pi].p[j];
                                        vp.x *= param_->vox_size ;
                                        vp.y *= param_->vox_size ;
                                        vp.z *= param_->vox_size ;

//                                        std::cout << vp.x << " " <<vp.y << std::endl;
                                        vertices.push_back(vp);
                                    } else{
                                        f.vIdx[j] = vertexHashTable[tri_[pi].p[j]];
                                    }
                                }
                                if(flag)
                                    faces.push_back(f);
                            }
                            if(flag)
                                validCount ++;
                        }
                    }

                }
            }
        }

        // for (size_t i = 0; i < param_->total_vox; ++i) {
        //     int zi = i / (param_->vox_dim.x * param_->vox_dim.y);
        //     int yi = (i - zi * param_->vox_dim.x * param_->vox_dim.y) / param_->vox_dim.x;
        //     int xi = i - zi * param_->vox_dim.x * param_->vox_dim.y - yi * param_->vox_dim.x;
        //     if (xi == param_->vox_dim.x - 1 || yi == param_->vox_dim.y - 1 || zi == param_->vox_dim.z - 1)
        //         continue;



        //     /* Create the triangle */
        //     for (int ti = 0; param_->triTable[cubeIndex][ti] != -1; ti += 3) {
        //         Face f;
        //         Triangle t;
        //         t.p[0] = vertlist[param_->triTable[cubeIndex][ti]];
        //         t.p[1] = vertlist[param_->triTable[cubeIndex][ti + 1]];
        //         t.p[2] = vertlist[param_->triTable[cubeIndex][ti + 2]];

        //         uint3 grid_size = make_uint3(param_->vox_dim.x, param_->vox_dim.y, param_->vox_dim.z);
        //         for (int pi = 0; pi < 3; ++pi) {
        //             int idx = find_vertex(t.p[pi], grid_size, param_->vox_size, hash_table);
        //             if (idx == -1) {
        //                 insert_vertex(t.p[pi], vertexCount, grid_size, param_->vox_size, hash_table);
        //                 f.vIdx[pi] = vertexCount++;
        //                 t.p[pi].x = t.p[pi].x * param_->vox_size + param_->vox_origin.x;
        //                 t.p[pi].y = t.p[pi].y * param_->vox_size + param_->vox_origin.y;
        //                 t.p[pi].z = t.p[pi].z * param_->vox_size + param_->vox_origin.z;
        //                 vertices.push_back(t.p[pi]);
        //             } else
        //                 f.vIdx[pi] = idx;
        //         }
        //         faces.push_back(f);
        //     }
        // }


        std::cout << vertices.size() << std::endl;
        std::ofstream plyFile;
        plyFile.open(outputFileName);
        plyFile << "ply\nformat ascii 1.0\ncomment stanford bunny\nelement vertex ";
        plyFile << vertices.size() << "\n";
        plyFile << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
        plyFile << "element face " << faces.size() << "\n";
        plyFile << "property list uchar int vertex_index\nend_header\n";
        for (auto v : vertices) {
            plyFile << v.x << " " << v.y << " " << v.z << " " << (int) v.r << " " << (int) v.g << " " << (int) v.b
                    << "\n";

//            plyFile << v.x << " " << v.y << " " << v.z << "\n";
        }
        for (auto f : faces) {
            plyFile << "3 " << f.vIdx[0] << " " << f.vIdx[1] << " " << f.vIdx[2] << "\n";
        }
        plyFile.close();
        std::cout << "File saved" << std::endl;
        std::cout << "totalsize = "<< totalsize << " valid = "<< validCount << std::endl;
    }

    __host__
    int3 GpuTsdfGenerator::calc_cell_pos(Vertex p, float cell_size) {
        int3 cell_pos;
        cell_pos.x = int(floor(p.x / cell_size));
        cell_pos.y = int(floor(p.y / cell_size));
        cell_pos.z = int(floor(p.z / cell_size));
        return cell_pos;
    }

    __host__
    unsigned int GpuTsdfGenerator::calc_cell_hash(int3 cell_pos, uint3 grid_size) {
        if (cell_pos.x < 0 || cell_pos.x >= (int) grid_size.x
            || cell_pos.y < 0 || cell_pos.y >= (int) grid_size.y
            || cell_pos.z < 0 || cell_pos.z >= (int) grid_size.z)
            return (unsigned int) 0xffffffff;

        cell_pos.x = cell_pos.x & (grid_size.x - 1);
        cell_pos.y = cell_pos.y & (grid_size.y - 1);
        cell_pos.z = cell_pos.z & (grid_size.z - 1);

        return ((unsigned int) (cell_pos.z)) * grid_size.y * grid_size.x
               + ((unsigned int) (cell_pos.y)) * grid_size.x
               + ((unsigned int) (cell_pos.x));
    }

    __host__
    int GpuTsdfGenerator::find_vertex(Vertex p, uint3 grid_size, float cell_size,
                                      std::vector<std::list<std::pair<Vertex, int>>> &hash_table) {
        unsigned int key = calc_cell_hash(calc_cell_pos(p, cell_size), grid_size);
        if (key != 0xffffffff) {
            std::list<std::pair<Vertex, int>> ls = hash_table[key];
            for (auto it = ls.begin(); it != ls.end(); ++it) {
                if ((*it).first.x == p.x && (*it).first.y == p.y && (*it).first.z == p.z) {
                    return (*it).second;
                }
            }
        }
        return -1;
    }

    __host__
    void GpuTsdfGenerator::insert_vertex(Vertex p, int index, uint3 grid_size, float cell_size,
                                         std::vector<std::list<std::pair<Vertex, int>>> &hash_table) {
        unsigned int key = calc_cell_hash(calc_cell_pos(p, cell_size), grid_size);
        if (key != 0xffffffff) {
            std::list<std::pair<Vertex, int>> ls = hash_table[key];
            for (auto it = ls.begin(); it != ls.end(); ++it) {
                if ((*it).first.x == p.x && (*it).first.y == p.y && (*it).first.z == p.z) {
                    (*it).second = index;
                    return;
                }
            }
            ls.push_back(std::make_pair(p, index));
        }
        return;
    }

    __host__
    std::vector<Vertex>* GpuTsdfGenerator::getVertices() {
        return &global_vertex;
    }

    __host__
    std::vector<Face>* GpuTsdfGenerator::getFaces() {
        return &global_face;
    }

    __host__
    std::vector<std::list<std::pair<Vertex, int>>>* GpuTsdfGenerator::getHashMap() {
        return &global_map;
    }

    __host__
    MarchingCubeParam* GpuTsdfGenerator::getMarchingCubeParam() {
        return param_;
    }


    //hashing function

    // __global__
    // void InsertHashGPU(int3 *keys,
    //     VoxelBlock *values,
    //     int n,
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> bm) {
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;

    //   if (base >= n) {
    //     return;
    //   }
    //   bm[keys[base]] = values[base];
    // }

    bool operator==(const VoxelBlock &a, const VoxelBlock &b) {
        for (int i=0; i<4*4*4; i++) {
            // if(&a.voxels[i] != &b.voxels[i])
            //     return false;
            if(a.voxels[i].sdf != b.voxels[i].sdf) {
                return false;
            }
            // if(a.voxels[i].sdf_color != b.voxels[i].sdf_color){
            //     return false;
            // }
            // if(a.voxels[i].weight != b.voxels[i].weight){
            //     return false;
            // }
        }
        return true;
    }

    // __global__
    // void kernel(int3 *keys,
    //     VoxelBlock *values,
    //     int n,
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> bm) {
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;

    //   if (base >= n) {
    //     return;
    //   }
    //   bm[keys[base]] = values[base];
    // }

    __device__
    float2 cameraToKinectScreenFloat(const float3& pos, MarchingCubeParam* param)   {
        //return make_float2(pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx, c_depthCameraParams.my - pos.y*c_depthCameraParams.fy/pos.z);
        return make_float2(
            pos.x*param->fx/pos.z + param->cx,            
            pos.y*param->fy/pos.z + param->cy);
    }

    __device__
    float cameraToKinectProjZ(float z, MarchingCubeParam* param)    {
        return (z - param->min_depth)/(param->max_depth - param->min_depth);
    }

    __device__
    float3 cameraToKinectProj(const float3& pos, MarchingCubeParam* param) {
        float2 proj = cameraToKinectScreenFloat(pos, param);

        float3 pImage = make_float3(proj.x, proj.y, pos.z);

        pImage.x = (2.0f*pImage.x - (param->im_width- 1.0f))/(param->im_width- 1.0f);
        //pImage.y = (2.0f*pImage.y - (c_depthCameraParams.m_imageHeight-1.0f))/(c_depthCameraParams.m_imageHeight-1.0f);
        pImage.y = ((param->im_height-1.0f) - 2.0f*pImage.y)/(param->im_height-1.0f);
        pImage.z = cameraToKinectProjZ(pImage.z, param);

        return pImage;
    }

    __device__ float3 wolrd2cam(float* c2w_, float3 pos, MarchingCubeParam* param){
                // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pos.x - c2w_[0 * 4 + 3];
        tmp_pt[1] = pos.y - c2w_[1 * 4 + 3];
        tmp_pt[2] = pos.z - c2w_[2 * 4 + 3];
        float pt_cam_x =
                c2w_[0 * 4 + 0] * tmp_pt[0] + c2w_[1 * 4 + 0] * tmp_pt[1] + c2w_[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y =
                c2w_[0 * 4 + 1] * tmp_pt[0] + c2w_[1 * 4 + 1] * tmp_pt[1] + c2w_[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z =
                c2w_[0 * 4 + 2] * tmp_pt[0] + c2w_[1 * 4 + 2] * tmp_pt[1] + c2w_[2 * 4 + 2] * tmp_pt[2];

        return make_float3(pt_cam_x, pt_cam_y, pt_cam_z);
    }

    __device__ bool isBlockInCameraFrustum(float3 blocks_pos, float* c2w, MarchingCubeParam* param){
        // return true;
        // printf("device print\n");
        float3 pCamera = wolrd2cam(c2w, blocks_pos,param);
        float3 pProj = cameraToKinectProj(pCamera, param);
        //pProj *= 1.5f;    //TODO THIS IS A HACK FIX IT :)
        pProj *= 0.95;
        return !(pProj.x < -1.0f || pProj.x > 1.0f || pProj.y < -1.0f || pProj.y > 1.0f || pProj.z < 0.0f || pProj.z > 1.0f); 
    }

    // __global__ void hashCopyKernel(vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks, 
    //     vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_){
    //     int base = blockDim.x * blockIdx.x  +  threadIdx.x;
    //     vhashing::HashEntryBase<int3> &entr = dev_blockmap_chunks.hash_table[base];
    //     VoxelBlock vb;
    //     dev_blockmap_[entr.key] = vb;//entr.value;
    // }

    __device__ volatile int sem = 0;

    __device__ void acquire_semaphore(volatile int *lock){
      while (atomicCAS((int *)lock, 0, 1) != 0)
        printf("wait\n");
    }

    __device__ void release_semaphore(volatile int *lock){
      // *lock = 0;
      // __threadfence();
        atomicExch((int*)lock, 0);
    }


    __global__ void HashAssignKernel(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w, 
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_chunks,
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_,
        unsigned int *d_heapBlockCounter, VoxelBlockPos* d_inBlockPosHeap){

        const unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * DDA_STEP;
        const unsigned int y = (blockIdx.y * blockDim.y + threadIdx.y) * DDA_STEP;
        if(x < width && y < height){
            // if(y == 0)
            //     printf("%d\n", x);

            float d = depth[y * width + x];
//            printf("param->maxdepth is %f\n", param->max_depth);

            if(d == 0.0f || d == MINF)
                return;

            if(d >= param->max_depth)
                return;

            float t = 10; // debug
            // printf("trunc_margin is %f\n\n",t);
            float minDepth = min(param->max_depth, d - t);
            // 新添加代码
            if(minDepth < 0)
                minDepth = 0;
            // minDepth = max(minDepth,0);
            float maxDepth = min(param->max_depth, d + t);
            if(minDepth >= maxDepth)
                return;
            // printf("min Depth is %f, max Depth is %f\n", minDepth, maxDepth);
            float3 raymin, raymax;
//            raymin = frame2base(x,y,param->min_depth, K, c2w, param);
//            raymax = frame2base(x,y,param->max_depth , K, c2w, param);
            raymin = frame2base(x,y,minDepth, K, c2w, param);
            raymax = frame2base(x,y,maxDepth , K, c2w, param);
            // float3 camcenter = frame2base(x,y,0, K, c2w, param);
            // printf("raymin\t%f\t%f\t%f\traymax \t%f\t%f\t%f\torigin\t%f\t%f\t%f\n", 
            //         raymin.x, raymin.y, raymin.z, raymax.x, raymax.y, raymax.z, camcenter.x, camcenter.y, camcenter.z);

            float3 rayDir = normalize(raymax - raymin);
            int3 idCurrentBlock = wolrd2block(raymin, param->block_size);
            int3 idEnd = wolrd2block(raymax, param->block_size);

            float3 cam_pos = make_float3(c2w[0 * 4 + 3], c2w[1 * 4 + 3], c2w[2 * 4 + 3]);

            // printf("kernel at(%d,%d), camera(%f,%f,%f), ray(%f,%f,%f) block(%d,%d,%d) to (%d,%d,%d)\n", x, y, cam_pos.x, cam_pos.y, cam_pos.z, rayDir.x, rayDir.y, rayDir.z,
            // idCurrentBlock.x, idCurrentBlock.y, idCurrentBlock.z,
            // idEnd.x, idEnd.y, idEnd.z);
            // DDA中的stepXYZ 判断当前的行进方向。
            float3 step = make_float3(sign(rayDir));
            int3 dirTem = make_int3(clamp(step, -1.0f, 1.0f));
            // 单纯判断 1, 0的方向。
//            if (step.x < 0 || step.y < 0 || step.z < 0) {
//                printf("step is %f %f %f\n dir is %d %d %d\n",
//                        step.x, step.y, step.z, dirTem.x, dirTem.y, dirTem.z);
//            }

            // 之前该位置的值为0.0f
            float3 boundarypos = block2world(idCurrentBlock + make_int3(clamp(step, -1.0f, 1.0f)), param->block_size) - 0.5f * param->vox_size;
            float3 tmax = (boundarypos - raymin) / rayDir;
            float3 tDelta = (step * param->vox_size * VOXEL_PER_BLOCK) / rayDir;
            int3 idBound = make_int3(make_float3(idEnd) + step);
            // 无法向前传播的错误情况
            if(rayDir.x == 0.0f || boundarypos.x - raymin.x == 0.0f){ tmax.x = PINF; tDelta.x = PINF;}
            if(rayDir.y == 0.0f || boundarypos.y - raymin.y == 0.0f){ tmax.y = PINF; tDelta.y = PINF;}
            if(rayDir.z == 0.0f || boundarypos.z - raymin.z == 0.0f){ tmax.z = PINF; tDelta.z = PINF;}

            unsigned int iter = 0;
            unsigned int maxLoopIterCount = 600;

            while(iter < maxLoopIterCount){
                float3 blocks_pos = block2world(idCurrentBlock, param->block_size);
                if(dev_blockmap_chunks.find(idCurrentBlock) != dev_blockmap_chunks.end()){
                    if(1 || isBlockInCameraFrustum(blocks_pos, c2w, param)){

                        int3 chunkpos = block2chunk(idCurrentBlock);
                        dev_blockmap_[idCurrentBlock] = dev_blockmap_chunks[idCurrentBlock];
                    }
                }
                if(tmax.x < tmax.y && tmax.x < tmax.z){
                    idCurrentBlock.x += step.x;
                    if(idCurrentBlock.x == idBound.x) {
                        return;
                    }
                    tmax.x += tDelta.x;
                }
                else if(tmax.z < tmax.y)
                {
                    idCurrentBlock.z += step.z;
                    if(idCurrentBlock.z == idBound.z) {
                        return;
                    }
                    tmax.z += tDelta.z;
                }
                else
                {
                    idCurrentBlock.y += step.y;
                    if(idCurrentBlock.y == idBound.y) {
                        return;
                    }
                    tmax.y += tDelta.y;
                }
                iter++;
            }
//            printf("Max Iterator is %d\n", iter);
        }
    }

    __host__ void GpuTsdfGenerator::HashReset(){
        // dev_block_idx.clear();
    }

    __global__ void getHeapCounterKernel(int* count, 
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> dev_blockmap_){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            *count = *(dev_blockmap_.heap_counter);
            // printf("count kernel 00 == %d \n", *count);
        }
            
    }

    __host__ void GpuTsdfGenerator::HashAssign(float *depth, const unsigned int height, const unsigned int width, 
        MarchingCubeParam *param, float* K, float* c2w){    
        // vhashing::HashTable<int3, VoxelBlock, BlockHasher, BlockEqual, vhashing::std_memspace>
        // bmhi_chunk(*dev_blockmap_chunks);
        {
            int dda_step = DDA_STEP;//光线每步所走的长度
            const dim3 grid_size((im_width_ / dda_step  + T_PER_BLOCK - 1) / T_PER_BLOCK, (im_height_ / dda_step + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
            const dim3 block_size(T_PER_BLOCK, T_PER_BLOCK, 1);//好像是在像素层面进行的并行化函数

            unsigned int dst = 0;
            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            // printf("start HashAssignKernel | d_heapBlockCounter\t%d\n", dst);

            HashAssignKernel <<< grid_size, block_size >>> (dev_depth_, im_height_, im_width_, dev_param_, dev_K_,
                    dev_c2w_, *dev_blockmap_chunks, *dev_blockmap_, d_heapBlockCounter, d_inBlockPosHeap);
            

            // hashCopyKernel <<< 100001, 4 >>> (*dev_blockmap_chunks, *dev_blockmap_);

            cudaSafeCall(cudaMemcpy(&dst, d_heapBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            // printf("Host: HashAssign d_heapBlockCounter %d\n", dst);
        }
        int *HeapCounter;
        cudaSafeCall(cudaMalloc((void**)&HeapCounter, sizeof(int)));
        
        {
            getHeapCounterKernel <<< 1, 1 >>> (HeapCounter, *dev_blockmap_);

        }

        // cudaSafeCall(cudaDeviceSynchronize()); //debug
        int dst_insert = 0;
        cudaMemcpy(&dst_insert, HeapCounter, sizeof(int), cudaMemcpyDeviceToHost);

        // unsigned int dst_u = dst_insert;
        cudaMemcpy(d_heapBlockCounter, &dst_insert, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // int src = 0;
        // cudaMemcpy(HeapCounter, &src, sizeof(int), cudaMemcpyHostToDevice);
        // printf("finish HashAssignKernel | d_heapBlockCounter\t%d\n", dst_insert);
        printf("headcounter is %d\n",dst_insert);
        cudaFree(HeapCounter);
    }
}



