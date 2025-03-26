#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <fstream>

using namespace nvinfer1;
using Clock = std::chrono::high_resolution_clock;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

ICudaEngine* loadEngine(const std::string& enginePath, ILogger& logger) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    delete runtime;

    return engine;
}


void printProgress(int current, int total) {
    const int barWidth = 40;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        std::cout << (i < pos ? '=' : ' ');
    }
    std::cout << "] " << int(progress * 100.0) << "% (" 
              << current << "/" << total << ")" << std::flush;
}


void measureTRT(const std::string& enginePath, int size, int iterations=100) {
    Logger logger;
    ICudaEngine* engine = loadEngine(enginePath, logger);
    if (!engine) {
        std::cerr << "Failed to load engine: " << enginePath << std::endl;
        return;
    }

    IExecutionContext* context = engine->createExecutionContext();
 
    std::string inputTensorName = engine->getIOTensorName(0);
    std::string outputTensorName = engine->getIOTensorName(1);

    int inputElems = 3 * size * size;
    int outputElems = 512 * size * size;
    size_t inputBytes = inputElems * sizeof(float);
    size_t outputBytes = outputElems * sizeof(float);

    // Allocate
    float* dummy = new float[inputElems];
    std::fill(dummy, dummy+inputElems, 0.5f);
    void* d_input; void* d_output;
    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_output, outputBytes);
    cudaMemcpy(d_input, dummy, inputBytes, cudaMemcpyHostToDevice);

    context->setTensorAddress(inputTensorName.c_str(), d_input);
    context->setTensorAddress(outputTensorName.c_str(), d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warm‑up
    for(int i=0;i<10;i++){
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }

    std::vector<double> times;
    times.reserve(iterations);
    for(int i=0;i<iterations;i++){
        auto t0 = Clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        printProgress(i+1, iterations);
    }
    std::cout << std::endl;

    double mean = std::accumulate(times.begin(), times.end(), 0.0)/iterations;
    double var = 0;
    for(double t: times) var += (t-mean)*(t-mean);
    var /= iterations;
    double stddev = std::sqrt(var);

    std::cout << "[RESULT] Size="<<size<<" Avg="<<mean<<" ms ± "<<stddev<<" ms\n";

    // Cleanup
    delete[] dummy;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
}



int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <engine.trt> <iterations> <size1> [size2 ...]\n";
        return -1;
    }

    std::string enginePath = argv[1];
    int iterations = std::stoi(argv[2]);
    for(int i = 3; i < argc; i++){
        int size = std::stoi(argv[i]);
        measureTRT(enginePath, size, iterations);
    }
    return 0;
}