#pragma once
#include <iostream>

struct Configuration
{
	float SegThreshold; // Segmentation Confidence Threshold(分割阈值)
	bool UseSingleMask;
	bool UseBoxInfo;
	std::string EncoderModelPath;
	std::string DecoderModelPath;
	std::string SaveDir;
	std::string Device;
};