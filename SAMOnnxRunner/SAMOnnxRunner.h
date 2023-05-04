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

class SAMOnnxRunner 
{
private:
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

	// CPU MemoryInfo
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
		OrtArenaAllocator, OrtMemTypeDefault
	);

	// Hardcode input node names
	unsigned int encoder_num_inputs = 2;
	std::vector<const char*> encoder_input_node_names;
	std::vector<std::vector<int64_t>> encoder_input_node_dims;
	unsigned int decoder_num_inputs = 2;
	std::vector<const char*> decoder_input_node_names;
	std::vector<std::vector<int64_t>> decoder_input_node_dims;

	// Hardcode output node names
	unsigned int encoder_num_outputs = 2;
	std::vector<const char*> encoder_output_node_names;
	std::vector<std::vector<int64_t>> encoder_output_node_dims;
	unsigned int decoder_num_outputs = 2;
	std::vector<const char*> decoder_output_node_names;
	std::vector<std::vector<int64_t>> decoder_output_node_dims;


protected:
	const unsigned int num_threads;

	cv::Mat Image_PreProcess(cv::Mat srcImage);
	void Encoder_PreProcess();
	void Decoder_PreProcess();

	void Encoder_BuildEmbedding();
	void Decoder_Inference();

	void Encoder_PostProcess();
	void Decoder_PostProcess();

	void InferenceSingleImage(Configuration cfg , cv::Mat srcImage);

	void InitOrtEnv(Configuration cfg);

public:
	explicit SAMOnnxRunner(unsigned int num_threads = 1);
	~SAMOnnxRunner();

	void setSegThreshold(float threshold);

	void ResetInitEncoder();

};
