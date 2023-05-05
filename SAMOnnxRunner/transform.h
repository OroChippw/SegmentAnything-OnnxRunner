#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

std::tuple<int , int> GetPreProcessShape(int old_h , int old_w , int long_side_length)
{
	double scale = long_side_length * 1.0 / MAX(old_h , old_w);
	int new_h = (int)(old_h * scale + 0.5);
	int new_w = (int)(old_w * scale + 0.5);
	std::tuple<int, int> newShape(new_h, new_w);
	return newShape;
}

cv::Mat ResizeLongestSide(int encoder_input_size, cv::Mat Image)
{
	cv::Mat resizeImage;
	const unsigned int h = Image.rows;
	const unsigned int w = Image.cols;
	std::tuple<int, int> newShape = GetPreProcessShape(h , w , encoder_input_size);

	//std::cout << std::get<0>(newShape) << std::endl;
	//std::cout << std::get<1>(newShape) << std::endl;

	cv::resize(Image , resizeImage , cv::Size(std::get<0>(newShape) , std::get<1>(newShape)) , cv::INTER_AREA);
	
	return resizeImage;
}
