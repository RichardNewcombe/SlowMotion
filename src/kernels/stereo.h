#ifndef CUDA_VISION_KERNEL_STEREOGEO_CU
#define CUDA_VISION_KERNEL_STEREOGEO_CU

#define getImageIdxStrided(x,y,stride)  ((y)*(stride) + (x))

#include "kernels.cu" //to get autocompletion and click through


texture<float, 2, cudaReadModeElementType> cudaTexFloatImage;
texture<float, 2, cudaReadModeElementType> cudaTexFloatkernel;
const static cudaChannelFormatDesc chandesc_float = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

__global__ void convolve2D(float * d_in, float * d_out, int2 imageSize, size_t imageStride,
                           float * d_kernel, int kernelSize, size_t kernelStride){

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int rad = kernelSize/2;

    float innerProduct=0;
    float normaliser = 0;


    for(int k = x-rad; k<=x+rad; k++){
        for(int j = y-rad; j<=y+rad; j++){

            if(k>=0 && k< imageSize.x && j>=0 && j< imageSize.y)
            {
                const float ker = d_kernel[(k-x+rad) + kernelStride*(j-y+rad)];
                normaliser+=ker;
                innerProduct += d_in[j*imageStride + k]*ker;
            }
        }
    }

    innerProduct/=normaliser;
    d_out[x + imageStride*y] = innerProduct;

}

__global__ void cuImageDivide(const  float * d_numerator,const  float * d_denominator, float * out, size_t imageStride){
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,imageStride);
    out[indexf1] = d_numerator[indexf1]/ d_denominator[indexf1];
}

__global__ void cuImageMultiply( float * d_a, float * d_b, float * out, size_t imageStride){
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexf1 = getImageIdxStrided(x,y,imageStride);
    out[indexf1] = d_a[indexf1]* d_b[indexf1];
}


__global__ void cuUCHAR4_FLOAT4(uchar4 * d_in,  float4 * d_out, size_t uchar4Stride,size_t float4Stride ){

    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int indexuchar4= getImageIdxStrided(x,y,uchar4Stride);
    const int indexfloat4= getImageIdxStrided(x,y,float4Stride);
    const uchar4 v = d_in[indexuchar4];
    d_out[indexfloat4] = make_float4(v.x/255.0f,v.y/255.0f,v.y/255.0f,1);
}

void char4_2_float4(uchar4 * d_in, float4 * d_out, int2 imageSize, size_t uchar4Pitch,size_t float4Pitch ){

    int wp = imageSize.x;
    int hp = imageSize.y;
    dim3 blockdim(boost::math::gcd<unsigned>(wp,16), boost::math::gcd<unsigned>(hp,16), 1);
    dim3 griddim( (wp) / blockdim.x, (hp) / blockdim.y);



    //predicted blur
    size_t uchar4Stride = uchar4Pitch/sizeof(uchar4);
    size_t float4Stride = float4Pitch/sizeof(float4);

    cuUCHAR4_FLOAT4<<<griddim,blockdim >>>(d_in,d_out,uchar4Stride,float4Stride);

}

void doRLiteration(float * d_originalBlured, float * d_unblured,
                   float * d_predictedBlur,float * d_poissonNoiseErrorTerm,  float * d_correctiveImage,
                   int2 imageSize, size_t imagePitch,
                   float * d_kernel, float * d_adjointKernel, int kernelSize, size_t kernelPitch){

    int wp = imageSize.x;
    int hp = imageSize.y;
    dim3 blockdim(boost::math::gcd<unsigned>(wp,16), boost::math::gcd<unsigned>(hp,16), 1);
    dim3 griddim( (wp) / blockdim.x, (hp) / blockdim.y);



    //predicted blur
    size_t imageStride = imagePitch/sizeof(float);
    size_t kernelStride = kernelPitch/sizeof(float) ;


    {

         ScopedCuTimer timer("convolve",true);
         convolve2D<<<griddim,blockdim >>>(d_unblured,d_predictedBlur,imageSize,imageStride,d_kernel,kernelSize,kernelStride);
    }

    {

         ScopedCuTimer timer("div",true);
        //noise error given Poisson model
        cuImageDivide<<<griddim,blockdim >>>(d_originalBlured,d_predictedBlur,d_poissonNoiseErrorTerm,imageStride);
    }

        //convolve with adjoint operator
        convolve2D<<<griddim,blockdim >>>(d_poissonNoiseErrorTerm,d_correctiveImage,imageSize,imageStride,d_adjointKernel,kernelSize,kernelPitch/sizeof(float) );

    {
             ScopedCuTimer timer("mul",true);
        //apply multiplicative update
        cuImageMultiply<<<griddim,blockdim >>>(d_correctiveImage,d_unblured,d_unblured,imageStride );
    }



}

#endif
