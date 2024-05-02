#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <filesystem>
#include <sstream>

#include <opencv2/opencv.hpp>
#include "inference.h"

#define IMAGE_FILE_TIME_FORMAT       "%F_%H-%M-%S"
#define CLASSES_FILE                 "classes.txt"
#define RUN_ON_GPU                   false
#define MODEL_INPUT_SHAPE            640, 480
#define MODEL_FILENAME               "yolov8s.onnx"

const std::filesystem::path PROJECT_DIR =       "../";
const std::filesystem::path MODEL_PATH =        PROJECT_DIR / MODEL_FILENAME;
const std::filesystem::path RESULT_PATH =       PROJECT_DIR / "results/";
const std::filesystem::path TEST_IMAGES_PATH =  PROJECT_DIR / "images/";


std::string GetFormattedTime()
{
    std::ostringstream oss;
    std::string formatstring = IMAGE_FILE_TIME_FORMAT;
    std::string time_string;

    auto time = std::time(nullptr);
    auto time_local = *std::localtime(&time);
    
    oss << std::put_time(&time_local, formatstring.c_str());
    
    time_string = oss.str();
    return time_string;
}

void DetectObjects(Inference &model, const std::filesystem::path &image)
{
    cv::Mat frame = cv::imread(image.string());

    std::vector<Detection> output = model.runInference(frame);

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;
        cv::rectangle(frame, box, color, 2);

        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    std::filesystem::path imgpath = RESULT_PATH / (GetFormattedTime() + ".jpg");
    cv::imwrite(imgpath.string(), frame);
}

void RunTests(Inference &inf)
{
    for (const std::filesystem::directory_entry & image : std::filesystem::directory_iterator(TEST_IMAGES_PATH))
    {
        DetectObjects(inf, image.path());
    }
}


int main(int argc, char** argv)
{
    std::filesystem::path ObjectDetectionModel = MODEL_PATH;
    cv::Size ModelInputShape{MODEL_INPUT_SHAPE};
    std::filesystem::path ClassesFile = PROJECT_DIR / CLASSES_FILE;
    bool RunOnGPU = RUN_ON_GPU;

    Inference InferenceModel(ObjectDetectionModel.string(), ModelInputShape, ClassesFile.string(), RunOnGPU);

    RunTests(InferenceModel);

    return 0;
}
