#include <string.h>
#include "dmain.h"


void via_dma_in0(float *in0, int len, float *data)
{
#pragma HLS INTERFACE axis port=data


   for(int i=0;i<len;i++)
    {
#pragma HLS PIPELINE
    	data[i] = in0[i];
    }
}







void via_dma_in1(float *in1, int len, float *weight)
{
#pragma HLS INTERFACE axis port=weight


   for(int i=0;i<len;i++)
    {
#pragma HLS PIPELINE
    	weight[i] = in1[i];
    }
}





void s2mm_data_copy(float *fifo, int len, float *buf)
{
#pragma HLS interface axis port=fifo
     for(int i=0; i<len; i++) {
#pragma HLS pipeline
          buf[i] = *fifo;
     }

}





void bgd(float *dain, int datalen, float *wtin, int weightlen, float *out)
{
	float hwdata[1], hwweight[1], hwgradient[1];
	//via_dma0(dain,datalen,daout);
	//via_dma1(wtin,weightlen,wtout);
	//bgd_ac(daout,wtout,graout);
	//dma_out(graout,weightlen,out);
	 via_dma_in0(dain,datalen,hwdata);
	 via_dma_in1(wtin,weightlen,hwweight);
	 bgd_accel(hwdata, hwweight, hwgradient);
	 s2mm_data_copy(hwgradient,weightlen , out);

}
