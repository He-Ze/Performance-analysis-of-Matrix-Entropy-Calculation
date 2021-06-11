/*
 * @Author: heze
 * @Date: 2021-06-01 00:38:55
 * @LastEditTime: 2021-06-05 00:31:05
 * @Description: CPU version
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;

#define printArray 0

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
    //分配矩阵和结果的内存
    int *host_array=(int*)malloc(sizeof(int)*size);
    float *host_result=(float*)malloc(sizeof(float)*size);
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
    clock_t start,end;
    start=clock();
    //计算
    for (int indexY = 0; indexY < height; indexY++){
        for (int indexX = 0; indexX < width; indexX++){
            int indexLeft = max(0, indexX-2);
            int indexRight = min(indexX+3, width);
            int indexUp = max(0, indexY-2);
            int indexDown = min(indexY+3, height);
            int indexNum = (indexRight-indexLeft) * (indexDown-indexUp);
            int indexTimes;
            float localResult = 0, indexP;
            for(int k=0;k<16;k++) {
                indexTimes = 0;
                for(int i=indexUp;i<indexDown;i++)
                    for(int j=indexLeft;j<indexRight;j++) {
                        if(host_array[i * width + j]==k){
                            indexTimes++;
                        }
                    }
                indexP = (float)indexTimes / indexNum;
                if(indexP!=0){
                    localResult -= indexP * log2(indexP);
                }
            }
            host_result[indexY*height+indexX] = localResult;
        }
    }
    end=clock();
    double time=(double)(end-start)/CLOCKS_PER_SEC;
    if(printArray){
        printf("结果：\n");
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                printf("%.5f ",host_result[i*width+j]);
            }
            printf("\n");
        } 
    }
    printf("矩阵维度%dx%d，在CPU上运行时间: %f ms.\n", height,width,time*1000);
    free(host_array);
    free(host_result);
}
