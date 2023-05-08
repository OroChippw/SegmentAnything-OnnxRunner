#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "SAMOnnxRunner.h"


std::tuple<int , int> GetPreProcessShape(int old_h , int old_w , int long_side_length)
{
	double scale = long_side_length * 1.0 / MAX(old_h , old_w);
	int new_h = (int)(old_h * scale + 0.5);
	int new_w = (int)(old_w * scale + 0.5);
	std::tuple<int, int> newShape(new_h, new_w);
	return newShape;
}

cv::Mat ResizeLongestSide_apply_image(cv::Mat Image , int encoder_input_size)
{
	cv::Mat resizeImage;
	const unsigned int h = Image.rows;
	const unsigned int w = Image.cols;
	std::tuple<int, int> newShape = GetPreProcessShape(h , w , encoder_input_size);

	cv::resize(Image , resizeImage , cv::Size(std::get<0>(newShape) , std::get<1>(newShape)) , cv::INTER_AREA);
	
	return resizeImage;
}

ClickInfo ResizeLongestSide_apply_coord(cv::Mat Image , ClickInfo clickinfo , int encoder_input_size)
{
	const unsigned int h = Image.rows;
	const unsigned int w = Image.cols;
	std::tuple<int, int> newShape = GetPreProcessShape(h, w, encoder_input_size);

	int new_w = std::get<0>(newShape);
	int new_h = std::get<1>(newShape);

	clickinfo.pt.x = clickinfo.pt.x * (new_w / w);
	clickinfo.pt.y = clickinfo.pt.y * (new_h / h);

	return clickinfo;
}

