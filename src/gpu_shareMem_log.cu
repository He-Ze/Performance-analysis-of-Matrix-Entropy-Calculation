/*
 * @Author: heze
 * @Date: 2021-06-01 00:38:55
 * @LastEditTime: 2021-06-05 00:47:39
 * @Description: 在gpu_shareMem.cu基础上查对数表
 * @FilePath: /src/gpu_shareMem_log.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define blockSize 10
#define printArray 0

/**
 * @brief 对数表
 */
__device__ float logTable[26]={0,
    0.000000000000000,1.000000000000000,1.584962500721156,2.000000000000000,2.321928094887362,
    2.584962500721156,2.807354922057604,3.000000000000000,3.169925001442312,3.321928094887362,
    3.459431618637297,3.584962500721156,3.700439718141092,3.807354922057604,3.906890595608519,
    4.000000000000000,4.087462841250339,4.169925001442312,4.247927513443585,4.321928094887363,
    4.392317422778761,4.459431618637297,4.523561956057013,4.584962500721156,4.643856189774724};

/**
 * @brief 核函数，在gpu_shareMem.cu基础上查对数表
 * 
 * @param width 矩阵列数
 * @param height 矩阵行数
 * @param array 待计算矩阵
 * @param globalResult 存放结果的矩阵
 * @return void
 */
__global__ void cal(int width, int height, int *array, float *globalResult) {
    //索引待计算元素位置
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int index = ix+iy*width;
    int indexX = index / width;
    int indexY = index % width;
    //计算需计算窗口的四条边分别是哪一行、哪一列，计算窗口总元素个数
    int indexLeft = max(0, indexY-2);
    int indexRight = min(indexY+3, width);
    int indexUp = max(0, indexX-2);
    int indexDown = min(indexX+3, height);
    int indexNum = (indexRight-indexLeft) * (indexDown-indexUp);

    //每个thread唯一的ID
    int shareID=threadIdx.y*blockSize+threadIdx.x;
    //分配共享内存
    __shared__ int indexTimes[16*10*10];
    float localResult = 0, indexP;

    for(int i=indexUp;i<indexDown;i++){
        for(int j=indexLeft;j<indexRight;j++) {
            indexTimes[shareID*16+array[i * width + j]]++;
        }
    }
    for(int i=0;i<16;i++){
        indexP = (float)indexTimes[i] / indexNum;
        if(indexP!=0.0){
            localResult -= indexP * (logTable[indexTimes[shareID*16+i]] - logTable[indexNum]);
        }
    }
    globalResult[index] = localResult;
}

/**
 * @description: 主函数
 * @param {int} argc  命令行参数个数
 * @param {char const} *argv 命令行参数指针
 * @return {*}
 */
int main(int argc, char const *argv[])
{
    //由运行时的命令行参数获取矩阵的行数和列数，并计算元素个数
    int height=atoi(argv[1]);
    int width=atoi(argv[2]);
    int size=height*width;
    int *host_array,*device_array;
    float *host_result,*device_result;
    //在CPU上分配矩阵和结果的内存
    cudaMallocHost((void **)&host_array,sizeof(int)*size);
    cudaMallocHost((void **)&host_result,sizeof(float)*size);
    //随机生成矩阵元素
    srand((unsigned)time(0));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            host_array[i * width + j] = rand()%16;
        }
    }
    if(printArray){
        printf("二维数组：\n");
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                printf("%2d ",host_array[i*width+j]);
            }
            printf("\n");
        } 
    }

    //在GPU上分配矩阵和结果的内存
    cudaMalloc((void **) &device_array, sizeof(int)*size);
    cudaMalloc((void **) &device_result, sizeof(float)*size);
    cudaMemcpy(device_array, host_array, sizeof(int)*size, cudaMemcpyHostToDevice);
    
    clock_t start,end;
    //分配线程块大小
    unsigned int grid_rows = (height/blockSize)+1;
    unsigned int grid_cols = (width/blockSize)+1;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(blockSize, blockSize);
    //调用核函数计算，并在前后计时，最后算出运行时间
    start=clock();
    cal<<<dimGrid, dimBlock>>>(width, height,device_array,device_result);
    cudaDeviceSynchronize();
    end=clock();
    double time_gpu=(double)(end-start)/CLOCKS_PER_SEC;
    //将结果从GPU拷贝回CPU，打印信息
    cudaMemcpy(host_result,device_result, sizeof(float)*size, cudaMemcpyDeviceToHost);
    if(printArray){
        printf("结果：\n");
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                printf("%.5f ",host_result[i*width+j]);
            }
            printf("\n");
        } 
    }
    printf("矩阵维度%dx%d，使用共享内存并查表在GPU上运行时间: %f ms.\n", height,width,time_gpu*1000);
    cudaFree(host_array);
    cudaFree(host_result);
    cudaFree(device_array);
    cudaFree(device_result);
}