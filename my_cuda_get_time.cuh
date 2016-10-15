#include <iostream>
#include <cstdio>
#include <string>
//+++++++++++++++++++++++++++++++++++
//  cuda event class
//   time -> "ms"(default) or "s"
//+++++++++++++++++++++++++++++++++++

#define __DEFAULT_CUDA_TIME_UNIT "ms"
#define __DEFAULT_CUDA_MAX_EVENT_NUM 10
class cudatimeStamp{
public:
    //Constructor
    cudatimeStamp()                           { initialize(__DEFAULT_CUDA_MAX_EVENT_NUM,__DEFAULT_CUDA_TIME_UNIT); }
    cudatimeStamp(int i)                      { initialize(i,__DEFAULT_CUDA_TIME_UNIT);}
    cudatimeStamp(int i,std::string time_unit){ initialize(i,time_unit); }
    //Destructor
    ~cudatimeStamp();
    //Get time
    void operator()(){cudaEventRecord(start[index++],0);syncflag=0;}
    void stamp()     {cudaEventRecord(start[index++],0);syncflag=0;}

    void sync();
    void setunit(std::string time_unit);
    float interval(int i,int j);
    float interval(int i);

    void print();
    friend std::ostream& operator<<(std::ostream& os, cudatimeStamp &cuts);
private:
    void initialize(int i,std::string time_unit);
	int limit;
    int index;
    int syncflag;
    cudaEvent_t *start;
    float       *elapsedTime;
    cudaEvent_t *s;
    float *e;
    std::string unit;//"ms"(default) or "s"
    float xrate;
};

cudatimeStamp::~cudatimeStamp(){
    delete start;
    delete elapsedTime;
}
void cudatimeStamp::initialize(int i,std::string time_unit){
    limit   =i;
    index   =0;
    syncflag=0;
    start = new cudaEvent_t [limit];
    setunit(time_unit);
    if(start==NULL){
        fprintf(stderr,"cudatimeStamp::allocation error\n");
        exit(1);
    }
    elapsedTime = new float [limit];

    if(elapsedTime==NULL){
        fprintf(stderr,"cudatimeStamp::allocation error\n");
        exit(1);
    }
	for(int i=0;i<limit;i++){
        cudaEventCreate(&start[i]);
        elapsedTime[i]=0.0;
	}
}
void cudatimeStamp::setunit(std::string time_unit){
    std::cout.precision(6);
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    if(time_unit=="s"){
        unit="s";
        xrate=0.0001;
    }else{
        unit="ms";
        xrate=1.0;
    }
}


void cudatimeStamp::sync(){
    cudaThreadSynchronize();
    if(syncflag) return;

	for(int i=0;i<index;i++)
	 	cudaEventSynchronize(start[i]);
	for(int i=0;i<index-1;i++)
	 	cudaEventElapsedTime(&elapsedTime[i], start[i], start[i+1]);
    syncflag=1;
    cudaError_t err=cudaGetLastError();
    if(err) printf("warning::cudatimeStamp::sync::%s(code:%d)\n",cudaGetErrorString(err),err);
}
float cudatimeStamp::interval(int i){
    if(syncflag==0) this->sync();
    return elapsedTime[i]*xrate;
}
float cudatimeStamp::interval(int i,int j){
    if(syncflag==0) this->sync();
    float v;
    cudaEventElapsedTime(&v,start[i],start[j]);
    return v*xrate;
}
void cudatimeStamp::print(){
    if(syncflag==0) this->sync();
	for(int i=0;i<index-1;i++)
        std::cout<<interval(i)<<","<<unit<<"\n";
}

std::ostream& operator<<(std::ostream& os, cudatimeStamp &cuts){
	for(int i=0;i<cuts.index-1;i++)
       std::cout<<cuts.interval(i)<<" , "<<cuts.unit<<" \n";
	return os;
}
