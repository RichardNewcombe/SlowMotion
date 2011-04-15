
#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/glcuda.h>
#include <vector_types.h>
#include <cvd/image_io.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>



using namespace CVD;
using namespace std;
using namespace pangolin;
#include <limits>

#include "utils.h"
#include "utilsShared.h"

//const std::string fname1 =  "/data/phd/workspace/code/svn/dataset/Aloe/view0.png";
//const std::string fname2 =  "/data/phd/workspace/code/svn/dataset/Aloe/view1.png";
const std::string fname1 =  "/data/phd/workspace/code/svn/dataset/heart/180608_Images/FullLeft000072.tiff";
const std::string fname2 =  "/data/phd/workspace/code/svn/dataset/heart/180608_Images/FullLeft000070.tiff";



void char4_2_float4(uchar4 * d_in, float4 * d_out, int2 imageSize, size_t uchar4Pitch,size_t float4Pitch );
float * getFlow( float * d_u, float * d_v, float * d_illum, int2 imageSize, size_t pitch  );

int main( int /*argc*/, char* argv[] )
{




    int gpu_id = cutGetMaxGflopsDeviceId();
    cudaGLSetGLDevice(gpu_id);

    //main2();

    pangolin::CreateGlutWindowAndBind("Main",640,480,GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL);
    glewInit();




    Image<Rgb<byte> > im0 =img_load(fname1);
    Image<Rgba<byte> > * imRGBA0 = new Image<Rgba<byte> >(im0.size()) ;
    Image<Rgb<byte> > *pVidFrame0 = &im0;
    convert_image(*pVidFrame0, *imRGBA0);

    Image<Rgb<byte> > im1 =img_load(fname2);
    Image<Rgba<byte> > * imRGBA1 = new Image<Rgba<byte> >(im0.size()) ;
    Image<Rgb<byte> > *pVidFrame1 = &im1;
    convert_image(*pVidFrame1, *imRGBA1);

    int2 imageSize = (int2){768,576 };//im0.size().x,im0.size().y};
    const int width =  imageSize.x;
    const int height = imageSize.y;


    size_t imagePitchFloat;
    float * d_u;
    float * d_v;
    float * d_illum;
    cutilSafeCall(cudaMallocPitch(&(d_u ),          &(imagePitchFloat),  imageSize.x* sizeof (float),  imageSize.y));
    cutilSafeCall(cudaMallocPitch(&(d_v ),          &(imagePitchFloat),  imageSize.x* sizeof (float),  imageSize.y));
    cutilSafeCall(cudaMallocPitch(&(d_illum ),          &(imagePitchFloat),  imageSize.x* sizeof (float),  imageSize.y));

    //cout << "real imageSIze " << im0.size() <<endl;
    float * test= getFlow( d_u, d_v, d_illum, imageSize,imagePitchFloat);


    // Create OpenGL window in single line thanks to GLUT
    //CreateGlutWindowAndBind("Main",640,480);

    float aspect = (float)imageSize.x/(float)imageSize.y;
    View& view_image0 = Display("image0").SetAspect(aspect);
    View& view_image1 = Display("image1").SetAspect(aspect);
    View& view_image2 = Display("image2").SetAspect(aspect);
    View& view_image3 = Display("image3").SetAspect(aspect);

    //cout <<  "Image size in " << im0.size().x << " " << im0.size().y << endl;





    View& d_panel = pangolin::CreatePanel("ui")
            .SetBounds(1.0, 0.0, 0, 150);


    View& d_imgs = pangolin::Display("images")
            //.SetBounds(1.0, 0.8, 150, 1.0, false)
            //first     distance from bottom left of opengl window i.e. 0.7 is 70%
            //co-ordinate from bottom left of screen from
            //0.0 to 1.0 for top, bottom, left, right.
            .SetBounds(1.0, 0.5, 150/*cus this is width in pixels of our panel*/, 1.0, false)
            .SetLayout(LayoutEqual)
            .AddDisplay(view_image0)
            .AddDisplay(view_image1)
            .AddDisplay(view_image2)
            .AddDisplay(view_image3)
            ;





    //GPU memory for grey scale images
    GlTexture greyTexture(width,height,GL_LUMINANCE32F_ARB);
    GlBufferCudaPtr greypbo( GlPixelUnpackBuffer, width*height*sizeof(float), cudaGraphicsMapFlagsNone,  GL_STREAM_DRAW  );

    //GPU memory for colour images
    GlTexture rgbaTexture(width, height, GL_RGBA32F);
    GlBufferCudaPtr rgbapbo(GlPixelUnpackBuffer,  width*height*sizeof(float4), cudaGraphicsMapFlagsNone, GL_STREAM_DRAW);



    size_t imagePitchUchar4;
    uchar4 * d_rgb_uchar4_temp;
    cutilSafeCall(cudaMallocPitch(&(d_rgb_uchar4_temp ),          &(imagePitchUchar4),  imageSize.x* sizeof (uchar4),  imageSize.y));



    //convert uchar4 to float4
    size_t imagePitchFloat4;
    float4 * d_rgb_float4_0;
    float4 * d_rgb_float4_1;


    //launch the kernel that just converts the uchar4 to float4
    cutilSafeCall(cudaMallocPitch(&(d_rgb_float4_0 ),          &(imagePitchFloat4),  imageSize.x* sizeof (float4),  imageSize.y));
    cutilSafeCall(cudaMallocPitch(&(d_rgb_float4_1 ),          &(imagePitchFloat4),  imageSize.x* sizeof (float4),  imageSize.y));

//    //uchar4 host to device image 0
//    cutilSafeCall(cudaMemcpy2D(d_rgb_uchar4_temp,imagePitchUchar4, (uchar4 *)imRGBA0->data(),imageSize.x*sizeof(uchar4),imageSize.x*sizeof(uchar4),imageSize.y,cudaMemcpyHostToDevice));
//    //convert
//    char4_2_float4(d_rgb_uchar4_temp,d_rgb_float4_0,  imageSize,imagePitchUchar4 , imagePitchFloat4 );

//    //uchar4 host to device image 1
//    cutilSafeCall(cudaMemcpy2D(d_rgb_uchar4_temp,imagePitchUchar4, (uchar4 *)imRGBA1->data(),imageSize.x*sizeof(uchar4),imageSize.x*sizeof(uchar4),imageSize.y,cudaMemcpyHostToDevice));
//    //convert
//    char4_2_float4(d_rgb_uchar4_temp,d_rgb_float4_1,  imageSize,imagePitchUchar4 , imagePitchFloat4 );


    // Default hooks for exiting (Esc) and fullscreen (tab).
    while(!pangolin::ShouldQuit())
    {

        static Var<bool> iterate("ui.Iterate",false,true);

        static Var<float> scale("ui.scale",0.1,0,1);

        glColor4f(1.0,1.0,1.0,1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(HasResized())
            DisplayBase().ActivateScissorAndClear();

        //if((iterate))
        {


            //DisplayFloatDeviceMem(&d_image,d_originalBlured, imagePitch,pbo_temp,imageTexture);
            //view_image0.Activate();
            //DisplayFloat4DeviceMem(test,imagePitchFloat4, rgbapbo,rgbaTexture);
//            view_image1.Activate();
//            DisplayFloat4DeviceMem(d_rgb_float4_1,imagePitchFloat4, rgbapbo,rgbaTexture);



            scaleGLPixels(make_float3(scale,scale,scale),(float3){0});
            view_image1.Activate();
            DisplayFloatDeviceMem(&view_image1, d_u,imagePitchFloat, greypbo,greyTexture);

            view_image2.Activate();
            DisplayFloatDeviceMem(&view_image2, d_v,imagePitchFloat, greypbo,greyTexture);

            view_image3.Activate();
            DisplayFloatDeviceMem(&view_image3, d_illum,imagePitchFloat, greypbo,greyTexture);

            scaleGLPixels(make_float3(1,1,1),(float3){0});
        }

        d_panel.Render();

        glutSwapBuffers();
        glutMainLoopEvent();
    }


    //cutilSafeCall(cudaFree(d_adjointkernel));

    return 0;
}



// system includes
#include <iostream>
#include <stdio.h>
#include <limits>


#include <cuda_runtime.h>
#include <iu/iucore.h>
#include <iu/iuio.h>
#include <iu/iumath.h>
#include <fl/flowlib.h>


float *  getFlow( float * d_u, float * d_v, float * d_illum, int2 imageSize, size_t pitch  )
{
    //cudaSetDevice(0); // set whatever device you wanna use...

    //  if(argc < 3)
    //  {
    //    std::cout << "usage: flowlib2_simpletest input_filename_1 input_filename_2" << std::endl;
    //    exit(EXIT_FAILURE);
    //  }

    // process parameters


    std::cout << "reading input images" << std::endl
              << "fixed image = " << fname1 << std::endl
              << "moving image = " << fname2 << std::endl;

    // Init FlowLib and set 2 images
    fl::FlowLib flow(0);
    bool ready = false;



    // read images using imageutilities
    iu::ImageGpu_32f_C1 *cu_im1 = iu::imread_cu32f_C1(fname1);
    iu::ImageGpu_32f_C1 *cu_im2 = iu::imread_cu32f_C1(fname2);

    //return (float *)cu_im1->data();




    flow.parameters().scale_factor = 0.95f;

    ready = flow.setInputImages(cu_im2, cu_im1);

    // parametrization
    flow.parameters().model = fl::HL1_TENSOR_ILLUMINATION_PRIMAL_DUAL;
    flow.parameters().iters = 10;
    flow.parameters().warps = 30;

    flow.parameters().lambda = 40.0f;
    flow.parameters().gamma_c =0.05f;
    //flow.parameters().epsilon_u = 0.1f;
    //  flow.parameters().verbose = 10;


//    if(!ready)
//        return EXIT_FAILURE;

    // do the calculations
    flow.calculate();

    // write flo result (see vision.middlebury.edu for more information of flo file format)
    flow.writeFloFile("simple_test_output.flo", 0);


    // use imageutilties
    IuSize result_size;
    flow.getSize(flow.parameters().stop_level, result_size);



    iu::ImageGpu_32f_C1 u(d_v, imageSize.x, imageSize.y, pitch, true);
    iu::ImageGpu_32f_C1 v(d_u, imageSize.x, imageSize.y, pitch, true);
    iu::ImageGpu_32f_C1 c(d_illum, imageSize.x, imageSize.y, pitch, true);
    //    iu::ImageGpu_8u_C4 cflow(result_size);


    flow.getU_32f_C1(flow.parameters().stop_level, &u);
    flow.getV_32f_C1(flow.parameters().stop_level, &v);
    flow.getIlluminationDifference_32f_C1(0, &c);
    //flow.getColorFlow_8u_C4(flow.parameters().stop_level, &cflow, 0.0f);

    return 0;
}
