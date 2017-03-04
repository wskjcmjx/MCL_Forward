# MCL_Forward
A thread safe re-implementation of [MathCoreLibrary](https://github.com/CompileSense/Temporary_MathCoreLibrary) forward propagation part.

This implementation, has been specifically optimized in response to **Intel CPU** and **Nvidia GPU** situation. As  for embedded/ mobile terminal situation, you may need [MCLdroid]().

##Overview
As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries will be used in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MKL-DNN](https://github.com/01org/mkl-dnn)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA8.0](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv5.1](https://developer.nvidia.com/cudnn)
- cuBLAS

##Features
  * Load model converted from `*.caffemodel`, model encrypt is supported.
  * Define layers' topology simply.
  * Supported layers currently: `INPUT`, `CONVOLUTION`, `POOLING`, `DENSE`(or `INNER_PRODUCT`), `RELU`.

##How to use
  * In `main.cpp` there is a example:
    1. First initialize a `CnnNet` object `net`, and then call `net.init('model', 'key')`, which will load the model named `model` and the blowfish key is `key`.
    2. Then call `net.forward('test.jpg', GRAY)`, which will read the file `test.jpg` in `GRAY` mode and do the net forward.
    3. Finally you can get the result and process it by yourself, or use `net.argmax()`. The function `argmax` is not really a argmax, and its result is not between 0 to 1, in fact, it will fetch all layers' max values whose `output` is defined as `true` and return them in vector.
    4. Since this example does a captcha recognition job, I call a simple function in `utils.cpp` to convert the numbers in the vector above to letters.
  * Define net's topology.
    1. In `CnnNet.cpp`, we can define net in `CnnNet::init`. Just `new` a `LayerConfig` and push_back it's address.
    2. If the `INPUT` layer's size(w, h) is set, all images will be resized when doing forward. Leave blank or set to `0` means pass the resize process.
    3. Some information is read from model, and here we don't need to define them, for example, the kernel size of convolution.
    4. When you don't set a layer's parent, it will be set to its previously pushed layer. If you want to set it, you can use string or vector to set it.
  * The model can be converted using `model_convertor.py`.

##Todo list
  * Change some layer implementation into MKL and  and CUDA implementation.
  * Separate the net's weights and the images calculated to make it threadsafe.
  * Support multi-machine, multi-uint(CPU) and multi-card(GPU)
  * Add more layer support, such as PReLU, Eltwise, Softmax...
  * Compress the model file.
  * More secure model encryption.
