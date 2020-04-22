#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "lrbgd.h"




//TRIPCOUNT identifier
const uint32_t gu32Chunks = NUM_CHUNKS;
const uint32_t gu32Classes = NUM_CLASSES;
const uint32_t gu32Features = (NUM_FEATURES + 1);
const uint32_t gu32Rows = (NUM_PARALLELERS * (NUM_CLASSES + NUM_FEATURES + 1));
const uint32_t gu32FactorNum = 5; //((NUM_FEATURES + 1)/157);




#ifdef _SDSVHLS_
void bgd_accel(value_t& valData, value_t& valWeight, value_t& valGradient, reg32_t regScalar)
{
#pragma HLS INTERFACE axis port=valData
#pragma HLS INTERFACE axis port=valWeight
#pragma HLS INTERFACE axis port=valGradient
#pragma HLS INTERFACE s_axilite register port=regScalar
#pragma HLS INTERFACE ap_ctrl_none port=return

    //float32_t fltData[NUM_PARALLELERS][NUM_TOTAL_FEATURE];
    float32_t fltData[NUM_KERNEL][NUM_TOTAL_FEATURE];
	#pragma HLS ARRAY_PARTITION variable=fltData dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltData1 cyclic factor=5
	//#pragma HLS ARRAY_PARTITION variable=fltData block factor=2 dim=2
    //float32_t fltData2[NUM_TOTAL_FEATURE];
	//#pragma HLS ARRAY_PARTITION variable=fltData2 dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltData2 cyclic factor=5



    //float32_t fltLabel[NUM_PARALLELERS][NUM_CLASSES];
    float32_t fltLabel[NUM_CLASSES];
	#pragma HLS ARRAY_PARTITION variable=fltLabel dim=1 complete
    //#pragma HLS ARRAY_PARTITION variable=fltLabel dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltLabel block factor=1

    float32_t fltWeight[NUM_CLASSES][NUM_TOTAL_FEATURE];
    #pragma HLS ARRAY_PARTITION variable=fltWeight dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltWeight cyclic factor=5 dim=1
	//#pragma HLS ARRAY_PARTITION variable=fltWeight cyclic factor=2 dim=2

    float32_t fltGradient[NUM_CLASSES][NUM_TOTAL_FEATURE];
    #pragma HLS ARRAY_PARTITION variable=fltGradient dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltGradient cyclic factor=5 dim=1
	//#pragma HLS ARRAY_PARTITION variable=fltGradient cyclic factor=2 dim=2
    uint32_t u32Chunks = regScalar;
    value_t valGTmp;

    valGTmp.data = 0;
    valGTmp.keep = 0xF;
    valGTmp.strb = 0;
    valGTmp.user = 0;
    valGTmp.last = 0;
    valGTmp.id = 0;
    valGTmp.dest = 0;

_RECV_WEIGHT_:
    for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_CLASSES * (NUM_FEATURES + 1)); loop++, j++)
    {
        #pragma HLS LOOP_TRIPCOUNT min=gu32Classes*gu32Features max=gu32Classes*gu32Features
        #pragma HLS PIPELINE II=1
        if(j == (NUM_FEATURES + 1))
        {
            i++;
            j = 0;
        }
        float32_t fltWTmp;
        value_t valWTmp = valWeight;
        fltWTmp.uint32 = valWTmp.data;
        fltWeight[i][j] = fltWTmp;
        fltWTmp.uint32 = 0;
        fltGradient[i][j] = fltWTmp;
        if(valWTmp.last == 1)
        {
            break;
        }
    }

_CHUNKS_LOOP_:
    for(uint32_t ch = 0; ch < NUM_CHUNKS; ch += NUM_PARALLELERS)
    {
_RECV_DATA_:
        for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_PARALLELERS * (NUM_CLASSES + NUM_FEATURES + 1)); loop++, j++)
        {
			#pragma HLS LOOP_TRIPCOUNT min=gu32Rows max=gu32Rows
			#pragma HLS PIPELINE II=1
			float32_t fltDTmp;
            value_t valDTmp = valData;
			if(j == (NUM_CLASSES + NUM_FEATURES + 1))
			{
				i++;
				j = 0;
			}
			fltDTmp.uint32 = valDTmp.data;
			if (j>=NUM_CLASSES)
			{
				for(uint32_t k = 0; k < NUM_KERNEL; k++)
					fltData[k][j - NUM_CLASSES] = fltDTmp;
			}
			else
			{
				fltLabel[j] = fltDTmp;
			}
        }

_GRADIENT_CALC_:
		for(uint32_t cla = 0; cla < NUM_CLASSES; cla+=NUM_KERNEL)
		{
			bgd_func0(fltLabel[cla], &fltWeight[cla][0], &fltData[0][0], &fltGradient[cla][0]);
			bgd_func1(fltLabel[cla+1], &fltWeight[cla+1][0], &fltData[1][0], &fltGradient[cla+1][0]);
			#if (NUM_KERNEL==5)
			bgd_func2(fltLabel[cla+2], &fltWeight[cla+2][0], &fltData[2][0], &fltGradient[cla+2][0]);
			bgd_func3(fltLabel[cla+3], &fltWeight[cla+3][0], &fltData[3][0], &fltGradient[cla+3][0]);
			bgd_func4(fltLabel[cla+4], &fltWeight[cla+4][0], &fltData[4][0], &fltGradient[cla+4][0]);
			#endif
		}
    }

_SEND_GRADIENT_:
    for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_CLASSES * (NUM_FEATURES + 1)); loop++, j++)
    {
        #pragma HLS LOOP_TRIPCOUNT min=gu32Classes*gu32Features max=gu32Classes*gu32Features
        #pragma HLS PIPELINE II=1
        if(j == (NUM_FEATURES + 1))
        {
            i++;
            j = 0;
        }
        if(loop == ((NUM_CLASSES * (NUM_FEATURES + 1)) - 1))
        {
        	valGTmp.last = 1;
        }
        float32_t fltGTmp;
        fltGTmp = fltGradient[i][j];
        valGTmp.data = fltGTmp.uint32;
        valGradient = valGTmp;
    }
}




void bgd_func0(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];



	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{
		#pragma HLS PIPELINE II=4
		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{
		#pragma HLS PIPELINE  II=1
		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}

void bgd_func1(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];
    //#pragma HLS ARRAY_PARTITION variable=fltDotTmp dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltDotTmp cyclic factor=5

	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{
		#pragma HLS PIPELINE II=4
		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{
		#pragma HLS PIPELINE  II=1
		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}

#if (NUM_KERNEL==5)
void bgd_func2(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];
    //#pragma HLS ARRAY_PARTITION variable=fltDotTmp dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltDotTmp cyclic factor=5

	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{
		#pragma HLS PIPELINE II=4
		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{
		#pragma HLS PIPELINE  II=1
		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}
void bgd_func3(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];
    //#pragma HLS ARRAY_PARTITION variable=fltDotTmp dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltDotTmp cyclic factor=5

	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{
		#pragma HLS PIPELINE II=4
		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{
		#pragma HLS PIPELINE  II=1
		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}
void bgd_func4(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];
    //#pragma HLS ARRAY_PARTITION variable=fltDotTmp dim=1 complete
	//#pragma HLS ARRAY_PARTITION variable=fltDotTmp cyclic factor=5

	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{
		#pragma HLS PIPELINE II=4
		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
		#pragma HLS UNROLL
		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{
		#pragma HLS PIPELINE  II=1
		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}
#endif

#else
void bgd_accel(float *valData, float *valWeight, float *valGradient)
{
#pragma HLS INTERFACE axis port=valData
#pragma HLS INTERFACE axis port=valWeight
#pragma HLS INTERFACE axis port=valGradient


	 uint32_t datapos=0;
	 float32_t fltData[NUM_KERNEL][NUM_TOTAL_FEATURE];

	 float32_t fltLabel[NUM_CLASSES];

	 float32_t fltWeight[NUM_CLASSES][NUM_TOTAL_FEATURE];


	 float32_t fltGradient[NUM_CLASSES][NUM_TOTAL_FEATURE];



_RECV_WEIGHT_:
    for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_CLASSES * (NUM_FEATURES + 1)); loop++, j++)
    {
       #pragma HLS LOOP_TRIPCOUNT min=gu32Classes*gu32Features max=gu32Classes*gu32Features
       #pragma HLS PIPELINE II=1
        if(j == (NUM_FEATURES + 1))
        {
            i++;
            j = 0;
        }
        float32_t fltWTmp;

		fltWTmp.float32 = valWeight[loop];
		fltWeight[i][j] = fltWTmp;
	    fltWTmp.uint32 = 0;
		fltGradient[i][j] = fltWTmp;





    }


_CHUNKS_LOOP_:
    for(uint32_t ch = 0; ch < NUM_CHUNKS; ch += NUM_PARALLELERS)
    {
_RECV_DATA_:
        for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_PARALLELERS * (NUM_CLASSES + NUM_FEATURES + 1)); loop++, j++,datapos++)
        {
            #pragma HLS LOOP_TRIPCOUNT min=gu32Rows max=gu32Rows
			#pragma HLS PIPELINE II=1
			float32_t fltDTmp;

			if(j == (NUM_CLASSES + NUM_FEATURES + 1))
			{
				i++;
				j = 0;
			}
			fltDTmp.float32 = valData[datapos];
			if (j>=NUM_CLASSES)
			{
				for(uint32_t k = 0; k < NUM_KERNEL; k++)
					fltData[k][j - NUM_CLASSES] = fltDTmp;
			}
			else
			{
				fltLabel[j] = fltDTmp;
			}
        }

_GRADIENT_CALC_:
		for(uint32_t cla = 0; cla < NUM_CLASSES; cla+=NUM_KERNEL)
		{
			bgd_func0(fltLabel[cla], &fltWeight[cla][0], &fltData[0][0], &fltGradient[cla][0]);
			bgd_func1(fltLabel[cla+1], &fltWeight[cla+1][0], &fltData[1][0], &fltGradient[cla+1][0]);
			#if (NUM_KERNEL==5)
			bgd_func2(fltLabel[cla+2], &fltWeight[cla+2][0], &fltData[2][0], &fltGradient[cla+2][0]);
			bgd_func3(fltLabel[cla+3], &fltWeight[cla+3][0], &fltData[3][0], &fltGradient[cla+3][0]);
			bgd_func4(fltLabel[cla+4], &fltWeight[cla+4][0], &fltData[4][0], &fltGradient[cla+4][0]);
			#endif
		}
    }

_SEND_GRADIENT_:
    for(uint32_t loop = 0, i = 0, j = 0; loop < (NUM_CLASSES * (NUM_FEATURES + 1)); loop++, j++)
    {
       #pragma HLS LOOP_TRIPCOUNT min=gu32Classes*gu32Features max=gu32Classes*gu32Features
       #pragma HLS PIPELINE II=1
        if(j == (NUM_FEATURES + 1))
        {
            i++;
            j = 0;
        }

        float32_t fltGTmp;
        fltGTmp = fltGradient[i][j];
        valGradient[loop]= fltGTmp.float32;

    }

}

void bgd_func0(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];



	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{

		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{

		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}

void bgd_func1(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];


	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{

		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{

		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}

#if (NUM_KERNEL==5)
void bgd_func2(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];


	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{

		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{

		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}
void bgd_func3(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];


	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{

		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{

		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}
void bgd_func4(float32_t vfltLabel, float32_t* pfltWeight, float32_t* pfltData, float32_t* pfltGradient)
{
    float32_t fltDotTmp[NUM_PIPEFACTOR];


	float32_t fltDif;
	float32_t fltDot;
	fltDot.float32 = 0.0f;

	_DOT_CALC0_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDotTmp[m].float32 = 0.0f;
	}

	_DOT_CALC_ALL_:
	for(int fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=NUM_PIPEFACTOR)
	{

		_DOT_CALC_LP:
		for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){
			fltDotTmp[m].float32 += pfltWeight[fe0+m].float32 * pfltData[fe0+m].float32;
		}
	}

	_DOT_CALC1_:
	for(uint32_t m = 0; m < NUM_PIPEFACTOR; m++){

		fltDot.float32 += fltDotTmp[m].float32;
	}

	_DIFF_CALC_:
	{
		fltDif.float32 = 1.0f / (1.0f + exp(0.0f - fltDot.float32)) - vfltLabel.float32;
	}

	_ACC_CALC_ALL_:
	for(uint32_t fe0 = 0; fe0 < NUM_TOTAL_FEATURE; fe0+=1)
	{

		pfltGradient[fe0].float32 += fltDif.float32 * pfltData[fe0].float32;
	}
}

#endif

#endif
