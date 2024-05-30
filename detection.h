#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <sstream>
#include "inference.h"

class ImageProcess
{
public:
	ImageProcess();
	std::vector<std::string> DetectObjects(Inference& model, const std::filesystem::path& image, bool SaveImages);
	void RunTests(void);
private:
	std::string FormatString;
	std::string GetFormattedTime(std::string FormatString);
	Inference InferenceModel;

	std::filesystem::path ProjectDir, ModelPath, ResultPath, TestImagesPath;
};

#endif