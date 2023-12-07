#pragma once
#include <iostream>

struct Configuration
{
	double SegThreshold; // Segmentation Confidence Threshold
	bool UseSingleMask;
	bool UseBoxInfo;
	bool KeepBoxInfo; // Reuse the same box information
	bool HasMaskInput = false; // Enter the existing mask into the decoder

	std::string EncoderModelPath;
	std::string DecoderModelPath;
	std::string SaveDir;
	std::string Device;
};