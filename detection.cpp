#include "detection.h"
#include "inference.h"

#define IMAGE_FILE_TIME_FORMAT  "%F_%H-%M-%S"
#define CLASSES_FILE            "classes.txt"
#define HAZARDS_FILE            "hazards.txt"
#define RUN_ON_GPU              false
#define MODEL_INPUT_SHAPE       640, 480
#define MODEL_FILENAME          "yolov8s.onnx"
#define SAVE_IMAGES             false

ImageProcess::ImageProcess()
{
    //  Initialize
    ProjectDir      = "../";
    ModelPath       = ProjectDir / MODEL_FILENAME;
    ResultPath      = ProjectDir / "results/";
    TestImagesPath  = ProjectDir / "images/";
    FormatString    = IMAGE_FILE_TIME_FORMAT;
    SaveImages      = SAVE_IMAGES;
    HazardsPath     = ProjectDir / HAZARDS_FILE;
    std::filesystem::path ClassesFile = ProjectDir / CLASSES_FILE;
    
    //  set inference model args
    std::filesystem::path ObjectDetectionModel = ModelPath;
    cv::Size ModelInputShape{ MODEL_INPUT_SHAPE };
    bool RunOnGPU = RUN_ON_GPU;

    //  Init inference model
    InferenceModel = Inference(ObjectDetectionModel.string(), ModelInputShape, ClassesFile.string(), RunOnGPU);

    //  Initialize hazards
    if(!std::filesystem::exists(ProjectDir / HAZARDS_FILE)) std::cout << "Hazard list does not exist, no hazard can be found." << std::endl;
    LoadHazardsFromFile();
}


std::vector<std::string> ImageProcess::DetectObjects(const std::filesystem::path& image)
{
    cv::Mat frame = cv::imread(image.string());

    std::vector<Detection> output = InferenceModel.runInference(frame);
    std::vector<std::string> DetectionList;

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    //  Object detection and outliner
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
    
    if (SaveImages)     // Save if needed
    {
        if (!std::filesystem::exists(ResultPath))
        {
            std::filesystem::create_directory(ResultPath);
        }
        int Duplicates = 1;
        std::string Filename = GetFormattedTime();
        while (std::filesystem::exists(ResultPath / (Filename + ".jpg")))
        {
            Filename = GetFormattedTime() + "_" + std::to_string(Duplicates++);
        }
        std::filesystem::path imgpath = ResultPath / (Filename + ".jpg");
        cv::imwrite(imgpath.string(), frame);
    }
    return DetectionList;
}

void ImageProcess::RunTests(void)
{
    std::vector<std::string> DetectionList;
    for (const std::filesystem::directory_entry& image : std::filesystem::directory_iterator(TestImagesPath))
    {
        DetectionList = DetectObjects(image.path());

        CheckForHazards(DetectionList);
    }
}

void ImageProcess::CheckForHazards(std::vector<std::string> &DetectionList)
{
    if (Hazards.size() == 0)
    {
        std::cout << "Empty hazard list, hazards can not be found." << std::endl;
        return;
    }
    for (int i = 0; i < DetectionList.size(); i++)
    {
        std::vector<std::string> DetectedHazards;
        for (int j = 0; j < Hazards.size(); j++)
        {
            //std::cerr << DetectionList[i] << " " << Hazards[j] << std::endl;
            if (DetectionList[i].compare(Hazards[j]) == 0)
            {
                DetectedHazards.push_back(Hazards[j]);
                break;
            }
        }
        if (DetectedHazards.size() != 0) SendAlert(DetectedHazards);
    }
}

void ImageProcess::SendAlert(std::vector<std::string> Hazards)
{
    for (int i = 0; i < Hazards.size(); i++) std::cout << "Hazard: " << Hazards[i] << " detected!" << std::endl;
    //  add an external alarm system handler here
}

void ImageProcess::LoadHazardsFromFile()
{
    std::ifstream inputFile(HazardsPath);
    if (inputFile.is_open())
    {
        std::string HazardLine;
        while (std::getline(inputFile, HazardLine, '\n'))
            Hazards.push_back(HazardLine);
        inputFile.close();
    }
}

std::string ImageProcess::GetFormattedTime()
{
    std::ostringstream oss;
    std::string time_string;

    time_t time = std::time(nullptr);
    tm time_local = *std::localtime(&time);

    oss << std::put_time(&time_local, FormatString.c_str());

    time_string = oss.str();
    return time_string;
}
     