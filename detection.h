#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <vector>
#include <filesystem>
#include <sstream>

#include "inference.h"

class ImageProcess
{
public:
	ImageProcess();
	std::vector<std::string> DetectObjects(const std::filesystem::path& image);
	void RunTests(void);
private:
	std::string FormatString;
	std::string GetFormattedTime();
	Inference InferenceModel;
	bool SaveImages;

	std::filesystem::path ProjectDir, ModelPath, ResultPath, TestImagesPath;
};

#endif