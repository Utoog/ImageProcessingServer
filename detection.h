#ifndef DETECTION_H
#define DETECTION_H

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <sstream>
#include <fstream>

#include "inference.h"

class ImageProcess
{
public:
	ImageProcess();
	std::vector<std::string> DetectObjects(const std::filesystem::path& image);
	void RunTests(void);
	void CheckForHazards(std::vector<std::string> &DetectionList);
	void SendAlert(std::vector<std::string> Hazards);
private:
	void LoadHazardsFromFile(void);
	std::string GetFormattedTime(void);
	std::filesystem::path ProjectDir, ModelPath, ResultPath, TestImagesPath, HazardsPath;
	std::vector<std::string> Hazards;
	std::string FormatString;
	Inference InferenceModel;
	bool SaveImages;
};

#endif