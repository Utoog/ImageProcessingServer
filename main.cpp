#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "inference.h"

//  yoinked from: https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-CPP-Inference

void StartInference(Inference &model, std::string &image)
{
    cv::Mat frame = cv::imread(image);

    // Inference starts here...
    std::vector<Detection> output = model.runInference(frame);

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    // Inference ends here...

    // This is only for preview purposes
    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    cv::imshow("Inference", frame);

    cv::waitKey(-1);
}

void RunTests(Inference &inf)
{
    std::vector<std::string> imageNames;
    imageNames.push_back("../images/image1.jpg");
    imageNames.push_back("../images/image2.jpg");
    imageNames.push_back("../images/image3.jpg");
    imageNames.push_back("../images/image4.jpg");

    for (int i = 0; i < imageNames.size(); i++)
    {
        StartInference(inf, imageNames[i]);
    }
}


int main(int argc, char** argv)
{
    std::string ObjectDetectionModel = "yolov8s.onnx";
    cv::Size ModelInputShape(640, 480);
    std::string ClassesFile = "classes.txt";
    bool RunOnGPU = false;

    Inference InferenceModel(ObjectDetectionModel, ModelInputShape, "", RunOnGPU);

    RunTests(InferenceModel);

    return 0;
}
