#pragma once

#include <opencv2/opencv.hpp>

std::tuple<int , int> GetPreProcessShape(int old_h , int old_w , int long_side_length)
{
	double scale = long_side_length ;
	


}

cv::Mat ResizeLongestSide(int encoder_input_size, cv::Mat Image)
{
	
	int h = Image.rows;
	int w = Image.cols;
	cv::Mat result;
	
	return Image;
}
