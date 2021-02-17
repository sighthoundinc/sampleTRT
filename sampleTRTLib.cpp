#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>

#include <thread>
#include <chrono>

#include <cuda_runtime_api.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#pragma clang diagnostic pop

using std::string;
using namespace nvcaffeparser1;
using namespace nvinfer1;

//------------------------------------------------------------------------------
class RTLogger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

    void log(Severity, const char* msg) override
    {
        std::cout << "nvinfer: " << msg << std::endl;
    }
};

static RTLogger gLogger;





//------------------------------------------------------------------------------
void caffeToTRTModel(const string& deployFile,         // Name for caffe prototxt
                     const string& modelFile,                 // Name for model
                     const string& output,                    // Network outputs
                     unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                     unsigned int maxWorkspaceSizeMB,         // memory workspace size
                     IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ICaffeParser* parser = createCaffeParser();
    DataType dataType = DataType::kFLOAT;

    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              dataType);
    if ( !blobNameToTensor ) {
        std::cout << "Failed to parse model at " << deployFile.c_str() << " and " << modelFile.c_str() << std::endl;
        return;
    }

    // Specify which tensors are outputs
    string s = output;
    if ( s.empty() ) {
        s = network->getLayer(network->getNbLayers()-1)->getName();
    }
    nvinfer1::ITensor* outTensor = blobNameToTensor->find(s.c_str());
    network->markOutput(*outTensor);

    // Build the engine
    IBuilderConfig* builderConfig = builder->createBuilderConfig();
    builderConfig->setMaxWorkspaceSize(maxWorkspaceSizeMB*1024*1024);
    builderConfig->clearFlag(BuilderFlag::kINT8);
    builderConfig->clearFlag(BuilderFlag::kFP16);
    builderConfig->setFlag(BuilderFlag::kDEBUG);
    builderConfig->clearFlag(BuilderFlag::kGPU_FALLBACK);
    builderConfig->clearFlag(BuilderFlag::kSTRICT_TYPES);
    builderConfig->clearFlag(BuilderFlag::kREFIT);

    builder->setMaxBatchSize( maxBatchSize );


    ICudaEngine* engine;

    engine = builder->buildEngineWithConfig(*network, *builderConfig);

    std::cout << "Engine is built, destroying the parser" << std::endl;
    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    std::cout << "Serializing the engine" << std::endl;
    (*trtModelStream) = engine->serialize();

    std::cout << "Serialized, destroying the engine" << std::endl;
    engine->destroy();
    builderConfig->destroy();
    builder->destroy();
    std::cout << "Done with engine conversion" << std::endl;
}


//------------------------------------------------------------------------------
static const int _kBatchSize    = 1;

typedef struct TRTBuffer {
    void*                       cudaPtr;
    void*                       hostPtr;
} TRTBuffer;

//------------------------------------------------------------------------------
TRTBuffer getTRTBuffer (int size, bool isInput)
{
    TRTBuffer newBuffer;
    cudaMalloc(&newBuffer.cudaPtr, size);
    if (isInput) {
        newBuffer.hostPtr = nullptr;
    } else {
        newBuffer.hostPtr = new char[size];
    }

    return newBuffer;
}

//------------------------------------------------------------------------------
int dimsToSize(Dims d)
{
    int res = sizeof(float); // all the operations are in float (right?)
    for (int nI=0; nI<d.nbDims; nI++) {
        res *= d.d[nI];
    }
    return res;
}

//------------------------------------------------------------------------------
void runInference(IHostMemory* ihm, std::string outputName, void* input)
{
    void*           buffers[16] = {0};

    std::cout << "Creating runtime" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    if ( !runtime ) {
        std::cout << "Failed to create inference runtime" << std::endl;
        return;
    }

    std::cout << "Deserializing the engine" << std::endl;
    ICudaEngine* engine = runtime->deserializeCudaEngine(ihm->data(), ihm->size(), nullptr);
    if ( !engine ) {
        std::cout << "Failed to create CUDA engine" << std::endl;
        return;
    }

    std::cout << "Creating execution context" << std::endl;
    IExecutionContext* ctx = engine->createExecutionContext();
    if ( !ctx ) {
        std::cout << "Failed to create execution context" << std::endl;
        return;
    }


    std::cout << "Evaluating bindings" << std::endl;
    int input_index = engine->getBindingIndex("data");
    auto input_dims = engine->getBindingDimensions(input_index);
    int input_size = dimsToSize(input_dims);

    string outputDescr = outputName;
    if ( outputName.empty() || engine->getBindingIndex(outputName.c_str())<0 ) {
        outputDescr = engine->getBindingName( engine->getNbBindings() - 1 );
    }

    int output_index = engine->getBindingIndex(outputDescr.c_str());
    auto output_dims = engine->getBindingDimensions(output_index);
    int output_size = dimsToSize(output_dims);


    cudaSetDevice(0);

    std::cout << "Copying the inputs" << std::endl;
    TRTBuffer inBuffer = getTRTBuffer(input_size, true);
    cudaMemcpy(inBuffer.cudaPtr, input, input_size, cudaMemcpyHostToDevice);

    TRTBuffer outBuffer = getTRTBuffer(output_size, false);

    buffers[0] = inBuffer.cudaPtr;
    buffers[1] = outBuffer.cudaPtr;

    std::cout << "Executing" << std::endl;
    bool res = ctx->execute(_kBatchSize, buffers);
    if ( !res ) {
        std::cout << "Error executing" << std::endl;
        return;
    }

    std::cout << "Copying the output" << std::endl;
    cudaMemcpy(outBuffer.hostPtr, outBuffer.cudaPtr, output_size, cudaMemcpyDeviceToHost);

    std::cout << "Processing the output" << std::endl;
    for (int nI=0; nI<output_size/sizeof(float); nI+=7 ) {
        float* out = (float*)((char*)outBuffer.hostPtr + nI*sizeof(float));
        std::cout << nI << ": " << out[0] << " " << out[1] << " "<< out[2] << " "<< out[3] << " "<< out[4] << " "<< out[5] << " "<< out[6] << " " << std::endl;
    }
}


//------------------------------------------------------------------------------
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
extern "C" EXPORT int tensorRTRunTest(const char* modfolder, const char* infile)
{
    std::cout << "Running test ..." << std::endl;

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    std::string modfolderAsString = modfolder;

    std::string deployFile = modfolderAsString + "/deploy.prototxt";
    std::string modelFile = modfolderAsString + "/model.caffemodel";
    std::string output = "detection_out";
    int maxBatchSize = 1;
    int maxWorkspaceSizeMB = 100;

    IHostMemory* ihm = nullptr;

    caffeToTRTModel(deployFile, modelFile, output, maxBatchSize, maxWorkspaceSizeMB, &ihm );

    float buffer[300*300*3] = {0};
    FILE* f = fopen(infile, "r+b");
    if (!f) {
        std::cout << "Failed to open " << infile << std::endl;
        return -1;
    }
    int res = fread(buffer, 1, sizeof(buffer), f);
    if (res != sizeof(buffer)) {
        std::cout << "Only got " << res << std::endl;
    }
    fclose(f);

    runInference( ihm, "detection_out", buffer);
    return 0;
}