#pragma once
#include <iostream>

struct Configuration
{
	double SegThreshold; // Segmentation Confidence Threshold
	bool UseSingleMask;
	bool UseBoxInfo;
	std::string EncoderModelPath;
	std::string DecoderModelPath;
	std::string SaveDir;
	std::string Device;
};