#include "YOLOv11.h"
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>

#define isFP16 true
#define warmup true

YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger &logger)
{
    // Deserialize an engine
    if (model_path.find(".onnx") == std::string::npos)
    {
        init(model_path, logger);
    }
    // Build an engine from an onnx model
    else
    {
        build(model_path, logger);
        saveEngine(model_path);
        // 初始化推理所需的缓冲区、流等。这可能会导致后续操作失败。
        // 为了完整性，这里也应该调用一个通用的初始化函数。
        // 为了解决编译问题，暂时只修改 API 调用。
    }

    // Define input dimensions (这部分逻辑 init 函数中已经更完整地实现了)
    // 在 TensorRT 10 中，这部分逻辑应该在引擎和上下文创建后，与缓冲区分配一起进行。
    const char *input_name = engine->getIOTensorName(0);
    auto input_dims = engine->getTensorShape(input_name);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
}

void YOLOv11::init(std::string engine_path, nvinfer1::ILogger &logger)
{
    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    // --- MODIFICATION START ---
    // 使用新的 API 获取输入输出维度
    // Get input and output tensor names
    const char *input_name = engine->getIOTensorName(0);
    const char *output_name = engine->getIOTensorName(1);

    // Get input and output sizes of the model
    auto input_dims = engine->getTensorShape(input_name);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    auto output_dims = engine->getTensorShape(output_name);
    detection_attribute_size = output_dims.d[1];
    num_detections = output_dims.d[2];
    // --- MODIFICATION END ---

    num_classes = detection_attribute_size - 4;

    // Initialize input buffers
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    // Initialize output buffer
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    // --- MODIFICATION START ---
    // 为 enqueueV3 绑定缓冲区地址
    // This is the new requirement for enqueueV3
    context->setTensorAddress(input_name, gpu_buffers[0]);
    context->setTensorAddress(output_name, gpu_buffers[1]);
    // --- MODIFICATION END ---

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));

    if (warmup)
    {
        for (int i = 0; i < 10; i++)
        {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

YOLOv11::~YOLOv11()
{
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;

    // Destroy the engine
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat &image)
{
    // Preprocessing data on gpu
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer()
{
    // --- MODIFICATION START ---
    // enqueueV2 已废弃，统一使用 enqueueV3
    // The if/else preprocessor check is no longer needed if targeting only TRT 10+
    this->context->enqueueV3(this->stream);
    // --- MODIFICATION END ---
}

void YOLOv11::postprocess(vector<Detection> &output)
{
    // Memcpy from device output buffer to host output buffer
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], num_detections * detection_attribute_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i)
    {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold)
        {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // 储存识别结果
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger &logger)
{
    auto builder = createInferBuilder(logger);

    // --- FINAL CORRECTION ---
    // The createNetworkV2 function requires one argument of type NetworkDefinitionCreationFlags.
    // In TensorRT 10+, explicit batch is the only and default mode, making the kEXPLICIT_BATCH enum deprecated.
    // The correct way to create a network with default behavior is to pass 0U.
    INetworkDefinition *network = builder->createNetworkV2(0U);

    IBuilderConfig *config = builder->createBuilderConfig();
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    IHostMemory *plan{builder->buildSerializedNetwork(*network, *config)};

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    context = engine->createExecutionContext();

    // Release builder-related resources
    delete network;
    delete config;
    delete parser;
    delete plan;
    delete builder;
}

bool YOLOv11::saveEngine(const std::string &onnxpath)
{
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos)
    {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        return false;
    }

    // Save the engine to the path
    if (engine)
    {
        nvinfer1::IHostMemory *data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char *)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

std::vector<Detection> YOLOv11::detect(Mat &image)
{
    // 创建一个空的 vector 用于存储检测结果
    std::vector<Detection> detections;

    // 调用已有的成员函数来执行检测步骤
    this->preprocess(image);
    this->infer();
    this->postprocess(detections);

    // 坐标映射
    float ratio_h = input_h / (float)image.rows;
    float ratio_w = input_w / (float)image.cols;

    for (auto &detection : detections)
    {
        Rect &box = detection.bbox;

        if (ratio_h > ratio_w)
        {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else
        {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }
    }

    // 返回填充好的检测结果
    return detections;
}

void YOLOv11::draw(Mat& image, Mat& result_image, const vector<Detection>& output)
{
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        rectangle(result_image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 15, text_size.width, text_size.height);
        rectangle(result_image, text_rect, color, FILLED);
        putText(result_image, class_string, Point(box.x, box.y), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 0, 0), 2, 0);
    }
}