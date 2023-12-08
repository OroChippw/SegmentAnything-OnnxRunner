#pragma once


#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "SAMConfig.h"


typedef struct BoxInfo 
{
	cv::Point left_top; // (x1,y1)为框选矩形左上角角点
	cv::Point right_bot; // (x2,y2)为框选矩形右下角角点

	BoxInfo() {};
	BoxInfo(int left_x, int left_y, int right_x, int right_y) {
		left_top.x = left_x;
		left_top.y = left_y;
		right_bot.x = right_x;
		right_bot.y = right_y;
	};
}BoxInfo;


struct ClickInfo
{
	cv::Point pt;
	bool positive;
};

struct MatInfo
{
	cv::Mat mask;
	float iou_pred;
};

class SAMOnnxRunner 
{
private:

	// Encoder Settings Params
	bool InitModelSession = false;
	bool InitEncoderEmbedding = false;
	int EncoderInputSize = 1024;

	// Decoder Settings Params
	double SegThreshold;

	// Env Settings Params
	std::string device{ "cpu" };
	Ort::Env env;
	Ort::SessionOptions session_options;
	std::unique_ptr<Ort::Session> EncoderSession, DecoderSession;
	std::vector<int64_t> EncoderInputShape, EncoderOutputShape;
	// CPU MemoryInfo and memory allocation
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault
	);

	const char* DecoderInputNames[6]{ "image_embeddings", "point_coords",   "point_labels",
							 "mask_input", "has_mask_input", "orig_im_size" },
		* DecoderOutputNames[3]{ "masks", "iou_predictions", "low_res_masks" };
	
	// Image Eembedding
	std::vector<float> image_embedding;

	// low_res_masks for mask_input
	std::vector<Ort::Value> tensors_buffer;


protected:
	const unsigned int num_threads;

	cv::Mat Image_PreProcess(cv::Mat srcImage);

	bool Encoder_BuildEmbedding(const cv::Mat& Image);
	
	std::vector<MatInfo> Decoder_Inference(Configuration cfg , cv::Mat srcIamge , ClickInfo clickinfo , BoxInfo boxinfo);


public:
	explicit SAMOnnxRunner(unsigned int num_threads = 1);
	~SAMOnnxRunner() {};

	void InitOrtEnv(Configuration cfg);

	std::vector<MatInfo> InferenceSingleImage(Configuration cfg, const cv::Mat& srcImage, ClickInfo clickInfo , BoxInfo boxinfo);

	void setSegThreshold(float threshold);

	void ResetInitEncoder();

};
