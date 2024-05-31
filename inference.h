#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#define CONFIDENCE_THRESHOLD    0.25
#define SCORE_THRESHOLD         0.45
#define NMS_THRESHOLD           0.50
#define LETTER_BOX_FOR_SQUARE   true

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    std::vector<std::string> classes;

    cv::Size2f modelShape{};

    float modelConfidenceThreshold { CONFIDENCE_THRESHOLD };
    float modelScoreThreshold      { SCORE_THRESHOLD };
    float modelNMSThreshold        { NMS_THRESHOLD };

    bool letterBoxForSquare = LETTER_BOX_FOR_SQUARE;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
