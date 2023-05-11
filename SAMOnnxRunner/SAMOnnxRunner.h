#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

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

typedef struct BoxInfo 
{
	int x1; // (x1,y1)为框选矩形左上角角点
	int y1;
	int x2; // (x2,y2)为框选矩形右下角角点
	int y2;
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
	// Image Eembedding
	//std::vector<Ort::Value> image_embedding;

	// Encoder Settings Params
	bool InitModelSession = false;
	bool InitEncoderEmbedding;
	int EncoderInputSize = 1024;

	// Decoder Settings Params
	float SegThreshold;

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
							 "mask_input",       "has_mask_input", "orig_im_size" },
		* DecoderOutputNames[3]{ "masks", "iou_predictions", "low_res_masks" };

	
	// input value handlers
	std::vector<float> input_bgr_value_handler;



	std::vector<float> image_embedding;

protected:
	const unsigned int num_threads;

	cv::Mat Image_PreProcess(cv::Mat srcImage);

	bool Encoder_BuildEmbedding(const cv::Mat& Image);
	
	std::vector<MatInfo> Decoder_Inference(cv::Mat srcIamge , ClickInfo clickinfo);




public:
	explicit SAMOnnxRunner(unsigned int num_threads = 1);
	~SAMOnnxRunner() {};

	void InitOrtEnv(Configuration cfg);

	void InferenceSingleImage(Configuration cfg, const cv::Mat& srcImage, ClickInfo clickInfo);

	void setSegThreshold(float threshold);

	void ResetInitEncoder();

};
