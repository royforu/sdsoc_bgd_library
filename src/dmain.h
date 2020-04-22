
//#define SIZE 4200000
//#pragma SDS data zero_copy(in0[0:SIZE])
//#pragma SDS data sys_port (in0:AFP)
//void via_dma_in0(float in0[SIZE], int len, float data[SIZE]);



void bgd_accel(float *valData, float *valWeight, float *valGradient);


#pragma SDS data zero_copy(in0[0:3180000])
#pragma SDS data sys_port (in0:HP)
#pragma SDS data access_pattern(in0:SEQUENTIAL)
#pragma SDS data mem_attribute(in0:PHYSICAL_CONTIGUOUS)
void via_dma_in0(float *in0, int len, float *data);

#pragma SDS data zero_copy(in1[0:10000])
#pragma SDS data sys_port (in1:HP)
#pragma SDS data access_pattern(in1:SEQUENTIAL)
#pragma SDS data mem_attribute(in1:PHYSICAL_CONTIGUOUS)
void via_dma_in1(float *in1, int len, float *weight);

#pragma SDS data zero_copy(buf[0:10000])
#pragma SDS data sys_port (buf:HP)
#pragma SDS data access_pattern(buf:SEQUENTIAL)
#pragma SDS data mem_attribute(buf:PHYSICAL_CONTIGUOUS)
void s2mm_data_copy(float *fifo, int len, float *buf);



