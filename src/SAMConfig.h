#pragma once
#include <iostream>

struct Configuration
{
	double SegThreshold = 0.9; // Segmentation Confidence Threshold
	bool UseSingleMask;
	bool UseBoxInfo;
	bool KeepBoxInfo = true; // Reuse the same box information
	bool HasMaskInput = false; // Enter the existing mask into the decoder

	std::string EncoderModelPath;
	std::string DecoderModelPath;
	std::string SaveDir;
	std::string Device;
};