#ifndef CUDA_VISION_UTILS_RL_CU
#define CUDA_VISION_UTILS_RL_CU
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>




inline  void DisplayFloatPBO(View* view, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    view->ActivateAndScissor();
    CopyPboToTex(pbo,tex_show, GL_LUMINANCE, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloat4PBO(View* view, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    view->ActivateAndScissor();
    CopyPboToTex(pbo,tex_show, GL_RGBA, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloat4PBO(GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    CopyPboToTex(pbo,tex_show, GL_RGBA, GL_FLOAT);
    tex_show.RenderToViewportFlipY();
}

inline void DisplayFloatDeviceMem(View* view, float* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {

        CudaScopedMappedPtr cu(pbo);
        //ScopedCuTimer t("DisplayFloatDeviceMem");
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float),  tex_show.height, cudaMemcpyDeviceToDevice));
    }
    DisplayFloatPBO(view, pbo, tex_show);
}

inline void DisplayFloat4DeviceMem(View* view, float4* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {
        CudaScopedMappedPtr cu(pbo);
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float4), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float4),  tex_show.height, cudaMemcpyDeviceToDevice));
    }


    DisplayFloat4PBO(view, pbo, tex_show);
}

inline void DisplayFloat4DeviceMem(float4* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
{
    {
        CudaScopedMappedPtr cu(pbo);
        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float4), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float4),  tex_show.height, cudaMemcpyDeviceToDevice));
    }
    DisplayFloat4PBO(pbo, tex_show);
}

inline void static scaleGLPixels(float3 scale, float3 bias){
    glPixelTransferf(GL_RED_SCALE, scale.x);
    glPixelTransferf(GL_GREEN_SCALE, scale.y);
    glPixelTransferf(GL_BLUE_SCALE, scale.z);
    glPixelTransferf(GL_RED_BIAS, bias.x);
    glPixelTransferf(GL_GREEN_BIAS, bias.y);
    glPixelTransferf(GL_BLUE_BIAS, bias.z);
}


//inline void DisplayFloatDeviceMem(float* d_ptr, size_t d_ptr_pitch, GlBufferCudaPtr& pbo, GlTexture& tex_show )
//{
//    {
//        CudaScopedMappedPtr cu(pbo);
//        cutilSafeCall(cudaMemcpy2D(*cu,tex_show.width*sizeof(float), d_ptr, d_ptr_pitch, tex_show.width * sizeof (float),  tex_show.height, cudaMemcpyDeviceToDevice));
//    }
//    DisplayFloatPBO(pbo, tex_show);
//}




inline void addUniformNoise(float * in,float mag, float *out, int2 imageSize){

    for(int i  =0 ; i <imageSize.x*imageSize.y ; i++){
        out[i] = in[i] +  mag*((float)rand()/RAND_MAX);
    }
}

inline void setRandomImageData(float * imageArray, int width, int height){
    for(int i = 0 ; i < width*height;i++) {
        imageArray[i] = (float)rand()/RAND_MAX;
    }
}

inline void imageFill(float * d, int2 imageSize, float val){
    for(int i = 0 ; i < imageSize.x*imageSize.y; i++) d[i]=val;
}

inline void imageNormalise(float * d, int2 imageSize, float min, float max){
    float minVal =0;
    float maxVal =0;
    for(int i = 0  ; i< imageSize.x*imageSize.y; i++) {
        minVal = fmin(d[i],minVal);
        maxVal = fmax(d[i],maxVal);
    }

    for(int i = 0  ; i< imageSize.x*imageSize.y; i++) {
        d[i] = (max-min)*((d[i]-minVal)/(maxVal-minVal)) +min;
    }
}

inline void createAdjoint(float * kernel, float * adjoint, int size){

    for(int x =  0 ;x < size; x++){
        for(int y = 0 ; y< size; y++){
            adjoint[y*size + x] = kernel[ (size-1-x) + (size-1-y)*size ];
            //            adjoint[x*size + y] = kernel[ (size-1-x) + (size-1-y)*size ];
        }
    }
}

inline void imageCopy(float * src, float * dst, int2 imageSize){
    for(int i =  0; i<imageSize.x*imageSize.y ; i++){
        dst[i] = src[i];
    }
}

inline void imageDivide(float * numerator, float * denominator, float * out, int2 imageSize){
    for(int i =  0; i<imageSize.x*imageSize.y ; i++){
        const float d = denominator[i];
        if(d!=0){
            out[i] = numerator[i]/d;
        }else{
            out[i]=0;
        }
    }
}

inline void imageSubtract(float * a, float * b, float * a_minus_b, int2 imageSize){
    for(int i =  0; i<imageSize.x*imageSize.y ; i++){
        a_minus_b[i] = a[i]-b[i];
    }
}




inline void convolve1D(float * dataIn,float * dataOut, int2 imageSize, float * kernel, int kernelSize, bool alongX){

    int rad = kernelSize/2;
    int stride = imageSize.x;
    float sum=0;

    for(int i = 0 ; i < kernelSize; i++)sum+=kernel[i];

    if(alongX){
        for(int x = rad; x< imageSize.x-rad; x++){
            for(int y = 0 ; y < imageSize.y ; y++){
                float innerProduct=0;
                for(int k = x-rad; k<=x+rad; k++){
                    innerProduct += dataIn[y*stride + k]*kernel[k-x+rad];
                }
                innerProduct/=sum;
                dataOut[x + stride*y] = innerProduct;
            }
        }
    }else{
        for(int x = 0 ; x < imageSize.x ; x++){
            for(int y = rad; y< imageSize.y-rad; y++){
                float innerProduct=0;
                for(int k = y-rad; k<=y+rad; k++){
                    innerProduct += dataIn[k*stride + x]*kernel[k-y+rad];
                }
                innerProduct/=sum;
                dataOut[x + stride*y] = innerProduct;
            }
        }
    }

}



#endif
