#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

using namespace std;

#define printArray 0

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

int main(int argc, char const *argv[])
{
    int height=atoi(argv[1]);
    int width=atoi(argv[2]);
    int size=height*width;
    int *host_array=(int*)malloc(sizeof(int)*size);
    float *host_result=(float*)malloc(sizeof(float)*size);
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

    float logTable[26]={0,
    0.000000000000000,1.000000000000000,1.584962500721156,2.000000000000000,2.321928094887362,
    2.584962500721156,2.807354922057604,3.000000000000000,3.169925001442312,3.321928094887362,
    3.459431618637297,3.584962500721156,3.700439718141092,3.807354922057604,3.906890595608519,
    4.000000000000000,4.087462841250339,4.169925001442312,4.247927513443585,4.321928094887363,
    4.392317422778761,4.459431618637297,4.523561956057013,4.584962500721156,4.643856189774724};

    double start,end;
    start=getTime();
    #pragma omp parallel for
    for (int indexY = 0; indexY < height; indexY++){
        for (int indexX = 0; indexX < width; indexX++){
            int indexLeft = max(0, indexY-2);
            int indexRight = min(indexY+3, width);
            int indexUp = max(0, indexX-2);
            int indexDown = min(indexX+3, height);
            int indexNum = (indexRight-indexLeft) * (indexDown-indexUp);
            int indexTimes[16]={0};
            float localResult = 0, indexP;
            for(int i=indexUp;i<indexDown;i++){
                for(int j=indexLeft;j<indexRight;j++) {
                    indexTimes[host_array[i * width + j]]++;
                }
            }
            for(int i=0;i<16;i++){
                indexP = (float)indexTimes[i] / indexNum;
                if(indexP!=0.0){
                    localResult -= indexP * (logTable[indexTimes[i]]-logTable[indexNum]);
                }
            }
            host_result[indexY*height+indexX] = localResult;
        }
    }
    end=getTime();
    double time=(end-start);
    if(printArray){
        printf("结果：\n");
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                printf("%.5f ",host_result[i*width+j]);
            }
            printf("\n");
        } 
    }
    printf("矩阵维度%dx%d，使用OpenMP在CPU上运行时间: %f ms.\n", height,width,time*1000);
    free(host_array);
    free(host_result);
}
