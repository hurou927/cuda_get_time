# cuda_get_time

cuda_get_time_header

Header file for getting time using cudaEvent

Include
======
```cpp
#include "my_cuda_get_time.cuh"
```

Initialize
======

```cpp
cudatimeStamp cutime();       //  defalut(10) times get-time (millisecond)
cudatimeStamp cutime(5);      //  5 times get-time (millisecond)
cudatimeStamp cutime(5,"s");  //  5 times get-time (second)
```
Stampe
======

```cpp
cutime.stamp();// or cutime();
//
// cuda code;
//
cutime.stamp();// or cutime();
```

Output
======

```cpp
std::cout<<cutime;   // print all interval-time
cutime.print();      // print all interval-time
cutime.interval(2,1);// 2st-time - 1st-time
```
