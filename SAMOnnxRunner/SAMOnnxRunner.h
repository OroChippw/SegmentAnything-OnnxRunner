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

class SAMOnnxRunner 
{
private:
	// Image Eembedding
	std::vector<Ort::Value> image_embedding;

	// Encoder Settings Params
	bool InitEncoder;
	int EncoderInputSize = 1024;

	// Decoder Settings Params
	float SegThreshold;

	// Env Settings Params
	std::string device{ "cpu" };
	Ort::Env env;
	Ort::SessionOptions session_options;
	Ort::Session *EncoderSession = nullptr;
	Ort::Session *DecoderSession = nullptr;

	// CPU MemoryInfo and memory allocation
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault
	);

	// Encoder Hardcode input and output node names
	unsigned int encoder_num_inputs;
	std::vector<const char*> encoder_input_node_names;
	std::vector<std::vector<int64_t>> encoder_input_node_dims;

	unsigned int encoder_num_outputs;
	std::vector<const char*> encoder_output_node_names;
	std::vector<std::vector<int64_t>> encoder_output_node_dims;

	// Decoder Hardcode input and output node names
	unsigned int decoder_num_inputs;
	std::vector<const char*> decoder_input_node_names;
	std::vector<std::vector<int64_t>> decoder_input_node_dims;

	unsigned int decoder_num_outputs;
	std::vector<const char*> decoder_output_node_names;
	std::vector<std::vector<int64_t>> decoder_output_node_dims;
	
	// input value handlers
	std::vector<float> input_bgr_value_handler;

	std::vector<Ort::AllocatedStringPtr>In_AllocatedStringPtr;
	std::vector<Ort::AllocatedStringPtr>Out_AllocatedStringPtr;
	/* 2023.05.05
	* TODO : 封装归一化过程
	*/

protected:
	const unsigned int num_threads;

	cv::Mat Image_PreProcess(cv::Mat srcImage);
	Ort::Value Encoder_PreProcess(cv::Mat Image);
	void Decoder_PreProcess(cv::Mat Image , ClickInfo clickinfo);

	std::vector<Ort::Value> Encoder_BuildEmbedding(Ort::Value* input_tensors);
	void Decoder_Inference();

	void Encoder_PostProcess();
	void Decoder_PostProcess();



public:
	explicit SAMOnnxRunner(unsigned int num_threads = 1);
	~SAMOnnxRunner();

	void InitOrtEnv(Configuration cfg);

	void InferenceSingleImage(Configuration cfg, cv::Mat srcImage, ClickInfo clickInfo);

	void setSegThreshold(float threshold);

	void ResetInitEncoder();

};
