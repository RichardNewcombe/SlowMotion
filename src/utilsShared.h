#ifndef CUDA_VISION_UTILS_RL_CU
#define CUDA_VISION_UTILS_RL_CU
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>



struct ScopedCuTimer{

		unsigned int timer;
		std::string name;
		bool doThreadSync;
		float last_time;


		inline float GetDeltaTime_s()
		{
			const float delta = cutGetTimerValue(timer) / 1000.0f;
			cutilCheckError(cutStopTimer(timer));
			cutilCheckError(cutResetTimer(timer));
			cutilCheckError(cutStartTimer(timer));
			return delta;
		}

		inline ScopedCuTimer(std::string name, bool doThreadSync=true):name(name),doThreadSync(doThreadSync){
				timer=0;
				last_time = 0;
				cutilCheckError( cutCreateTimer( &timer));
				if(doThreadSync) cudaThreadSynchronize();
				cutilCheckError(cutStartTimer(timer));
		};
		inline ~ScopedCuTimer(){
				if(doThreadSync) cudaThreadSynchronize();
				cutilCheckError(cutStopTimer(timer));
				std::cout<<  name << "::" << (float)cutGetTimerValue(timer)<< "ms" <<std::endl;
				cutilCheckError(cutDeleteTimer(timer));
		}
};




#endif
