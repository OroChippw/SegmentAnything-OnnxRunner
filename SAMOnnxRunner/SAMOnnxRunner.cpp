#pragma once

#include <iostream>
#include<Windows.h>
#include <chrono>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "SAMOnnxRunner.h"

wchar_t* multi_Byte_To_Wide_Char(std::string& pKey)
{
	// string 转 char*
	const char* pCStrKey = pKey.c_str();
	// 第一次调用返回转换后的字符串长度，用于确认为wchar_t*开辟多大的内存空间
	int pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
	wchar_t* pWCStrKey = new wchar_t[pSize];
	// 第二次调用将单字节字符串转换成双字节字符串
	MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
	// 不要忘记在使用完wchar_t*后delete[]释放内存
	return pWCStrKey;
}

cv::Mat SAMOnnxRunner::Image_PreProcess(cv::Mat srcImage)
{
	cv::Mat GrayImage;
	cv::cvtColor(srcImage, GrayImage, cv::COLOR_BGR2GRAY);
	return GrayImage;
}

void SAMOnnxRunner::Encoder_PreProcess() 
{
	auto start_time = std::chrono::steady_clock::now();
	auto end_time = std::chrono::steady_clock::now();
	std::cout << "Encoder build embedding cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
	 
}

void SAMOnnxRunner::Decoder_PreProcess()
{

}

void SAMOnnxRunner::Encoder_BuildEmbedding()
{

}

void SAMOnnxRunner::Decoder_Inference()
{

}

void SAMOnnxRunner::Encoder_PostProcess()
{

}

void SAMOnnxRunner::Decoder_PostProcess()
{

}

void SAMOnnxRunner::InferenceSingleImage(Configuration cfg , cv::Mat srcImage)
{
	cv::Mat Image = this->Image_PreProcess(srcImage);
	if (!this->InitEncoder)
	{
		this->Encoder_PreProcess();
	}

}

void SAMOnnxRunner::InitOrtEnv(Configuration cfg) 
{
	// 初始化OnnxRuntime运行环境
	this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SegmentAnythingModel");
	this->session_options = Ort::SessionOptions();
	this->session_options.SetInterOpNumThreads(num_threads);
	// 设置图像优化级别
	this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	this->session_options.SetLogSeverityLevel(4);

	// 初始化分割阈值、运行设备
	this->SegThreshold = cfg.SegThreshold;
	this->device = cfg.Device;

	// 若运行设备为GPU则开启CUDA加速，解开下述注释
	//if (device == "cuda") {
	//	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options , 0);
	//}

#if _WIN32
	const wchar_t* encoder_modelpath = multi_Byte_To_Wide_Char(cfg.EncoderModelPath);
	const wchar_t* decoder_modelpath = multi_Byte_To_Wide_Char(cfg.DecoderModelPath);
#else
	const char* encoder_modelpath = cfg.EncoderModelPath;
	const char* decoder_modelpath = cfg.DecoderModelPath;
#endif // _WIN32

	// 创建Session并加载模型进内存
	std::cout << "Using Onnxruntime C++ API" << std::endl;
	std::cout << "Building SegmentAnything Model Encoder ... " << std::endl;
	this->EncoderSession = new Ort::Session(env , encoder_modelpath , session_options);
	std::cout << "Building SegmentAnything Model Decoder ... " << std::endl;
	this->DecoderSession = new Ort::Session(env, decoder_modelpath, session_options);

	this->allocator = Ort::AllocatorWithDefaultOptions();

	delete encoder_modelpath;
	delete decoder_modelpath;
}

void SAMOnnxRunner::setSegThreshold(float threshold)
{
	this->SegThreshold = threshold;
}

void SAMOnnxRunner::ResetInitEncoder() 
{
	this->InitEncoder = false;
}

SAMOnnxRunner::SAMOnnxRunner(unsigned int threads) :
	num_threads(threads)
{
}

SAMOnnxRunner::~SAMOnnxRunner()
{
	if (EncoderSession)
	{
		delete EncoderSession;
	}
	EncoderSession = nullptr;
	if (DecoderSession)
	{
		delete DecoderSession;
	}
	DecoderSession = nullptr;
}

