#include "detection.h"
#include "inference.h"

#define IMAGE_FILE_TIME_FORMAT  "%F_%H-%M-%S"
#define CLASSES_FILE            "classes.txt"
#define RUN_ON_GPU              false
#define MODEL_INPUT_SHAPE       640, 480
#define MODEL_FILENAME          "yolov8s.onnx"
#define SAVE_IMAGES             true

ImageProcess::ImageProcess()
{
    ProjectDir = "../";
    ModelPath = ProjectDir / MODEL_FILENAME;
    ResultPath = ProjectDir / "results/";
    TestImagesPath = ProjectDir / "images/";
    FormatString = IMAGE_FILE_TIME_FORMAT;
    
    std::filesystem::path ObjectDetectionModel = ModelPath;
    cv::Size ModelInputShape{ MODEL_INPUT_SHAPE };
    std::filesystem::path ClassesFile = ProjectDir / CLASSES_FILE;
    bool RunOnGPU = RUN_ON_GPU;

    InferenceModel = Inference(ObjectDetectionModel.string(), ModelInputShape, ClassesFile.string(), RunOnGPU);

}

std::string ImageProcess::GetFormattedTime(std::string FormatString)
{
    std::ostringstream oss;
    //std::string FormatString = IMAGE_FILE_TIME_FORMAT;
    std::string time_string;

    auto time = std::time(nullptr);
    auto time_local = *std::localtime(&time);

    oss << std::put_time(&time_local, FormatString.c_str());

    time_string = oss.str();
    return time_string;
}

std::vector<std::string> ImageProcess::DetectObjects(Inference& model, const std::filesystem::path& image, bool SaveImages)
{
    cv::Mat frame = cv::imread(image.string());

    std::vector<Detection> output = model.runInference(frame);
    std::vector<std::string> DetectionList;

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];
        DetectionList.push_back(detection.className);

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
    if (SaveImages)
    {
        std::filesystem::path imgpath = ResultPath / (GetFormattedTime(FormatString) + ".jpg");
        cv::imwrite(imgpath.string(), frame);
    }
    return DetectionList;
}

void ImageProcess::RunTests(void)
{
    std::vector<std::string> DetectionList;
    for (const std::filesystem::directory_entry& image : std::filesystem::directory_iterator(TestImagesPath))
    {
        DetectionList = DetectObjects(InferenceModel, image.path(), SAVE_IMAGES);
    }
}