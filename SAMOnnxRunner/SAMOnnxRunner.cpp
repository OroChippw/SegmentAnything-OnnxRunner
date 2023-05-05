#pragma once

#include <iostream>
#include<Windows.h>
#include <chrono>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "SAMOnnxRunner.h"
#include "transform.h"

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
	std::cout << "PreProcess Image ..." << std::endl;
	cv::Mat rgbImage;
	cv::cvtColor(srcImage, rgbImage, cv::COLOR_BGR2RGB);
	return rgbImage;
}

Ort::Value SAMOnnxRunner::Encoder_PreProcess(cv::Mat Image) throw (std::runtime_error)
{
	// => Preprocess for encoder
	// Meta AI training encoder with a resolution of 1024 * 1024
	encoder_input_node_dims = {
		{1,3,1024,1024} // src after ResizeLongestSide , (b = 1 , c , h ,w)
	};

	const unsigned int channels = Image.channels();
	

	std::cout << "Transforms image by ResizeLongestSide ... " << std::endl;
	cv::Mat resizeImage = ResizeLongestSide(EncoderInputSize , Image);
	
	 // Normalization
	resizeImage.convertTo(resizeImage, CV_32FC3, 1.0f / 255.0f, 0.f);
	
	int pad_h = EncoderInputSize - resizeImage.rows;
	int pad_w = EncoderInputSize - resizeImage.cols;

	cv::Mat paddingImage;
	cv::copyMakeBorder(resizeImage , paddingImage , 0 ,pad_h , 0 , pad_w , cv::BorderTypes::BORDER_CONSTANT , cv::Scalar(0,0,0));
	
	// 构造输入Encoder的tensor
	const unsigned int target_channel = 3;
	const unsigned int target_height = 1024;
	const unsigned int target_width = 1024;
	const unsigned int target_tensor_size = target_channel * target_height * target_width;
	if (target_channel != channels)
	{
		throw std::runtime_error("Runtime Error Channel mismatch.");
	}

	std::cout << "Resize input value handler to "<< target_tensor_size << std::endl;
	input_bgr_value_handler.resize(target_tensor_size);

	std::vector<cv::Mat> mat_channels;
	cv::split(paddingImage, mat_channels);

	// C * H * W
	for (unsigned int i = 0; i < channels; i++)
	{
		std::memcpy(input_bgr_value_handler.data() + i * (target_height * target_width),
			mat_channels.at(i).data, target_height * target_width * sizeof(float));
	}
	std::cout << "Create Encoder input tensor ... " << std::endl;
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info_handler , input_bgr_value_handler.data() ,
		target_tensor_size , encoder_input_node_dims.at(0).data() ,
		encoder_input_node_dims.at(0).size());
	
	assert(input_tensor.IsTensor());

	return input_tensor;
}


Ort::Value SAMOnnxRunner::Encoder_BuildEmbedding(Ort::Value input_tensor)
{
	auto start_time = std::chrono::steady_clock::now();
	//this->EncoderSession->Run(
	//	Ort::RunOptions{nullptr} , 
	//);
	auto end_time = std::chrono::steady_clock::now();
	std::cout << "Encoder build embedding cost time : " << \
		std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
	return input_tensor;
}

void SAMOnnxRunner::Decoder_PreProcess()
{

}

void SAMOnnxRunner::Decoder_Inference()
{
	auto start_time = std::chrono::steady_clock::now();
	//this->DecoderSession->Run();
	auto end_time = std::chrono::steady_clock::now();
	std::cout << "Encoder build embedding cost time : " << \
		std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
}

void SAMOnnxRunner::Encoder_PostProcess()
{

}

void SAMOnnxRunner::Decoder_PostProcess()
{

}

void SAMOnnxRunner::InferenceSingleImage(Configuration cfg , cv::Mat srcImage , ClickInfo clickInfo)
{
	cv::Mat rgbImage = Image_PreProcess(srcImage);
	if (!InitEncoder)
	{
		std::cout << "InitEncoder is false , Preprocess before encoder image embedding ... " << std::endl;
		Ort::Value encoder_input_tensor = Encoder_PreProcess(rgbImage);
		Ort::Value image_embeddings = Encoder_BuildEmbedding(std::move(encoder_input_tensor));
		InitEncoder = true;
	}

}

void SAMOnnxRunner::InitOrtEnv(Configuration cfg) throw (std::runtime_error)
{
	// 初始化OnnxRuntime运行环境
	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SegmentAnythingModel");
	session_options = Ort::SessionOptions();
	session_options.SetInterOpNumThreads(num_threads);
	// 设置图像优化级别
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	session_options.SetLogSeverityLevel(4);

	// 初始化分割阈值、运行设备
	SegThreshold = cfg.SegThreshold;
	device = cfg.Device;

	// 若运行设备为GPU则开启CUDA加速，解开下述注释
	//if (device == "cuda") {
	//	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options , 0);
	//}

#if _WIN32
	std::cout << "Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
	const wchar_t* encoder_modelpath = multi_Byte_To_Wide_Char(cfg.EncoderModelPath);
	const wchar_t* decoder_modelpath = multi_Byte_To_Wide_Char(cfg.DecoderModelPath);
#else
	const char* encoder_modelpath = cfg.EncoderModelPath;
	const char* decoder_modelpath = cfg.DecoderModelPath;
#endif // _WIN32

	// 创建Session并加载模型进内存
	std::cout << "=> Using Onnxruntime C++ API" << std::endl;
	std::cout << "Building SegmentAnything Model Encoder ... " << std::endl;
	EncoderSession = new Ort::Session(env , encoder_modelpath , session_options);
	std::cout << "Building SegmentAnything Model Decoder ... " << std::endl;
	DecoderSession = new Ort::Session(env, decoder_modelpath, session_options);

	allocator = Ort::AllocatorWithDefaultOptions();

	std::cout << "=> Build EncoderSession and DecoderSession successfully." << std::endl;

	// Get Session input and output info
	encoder_num_inputs = EncoderSession->GetInputCount(); 	// Encoder
	encoder_num_outputs = EncoderSession->GetOutputCount();

	decoder_num_inputs = DecoderSession->GetInputCount(); // Decoder
	decoder_num_outputs = DecoderSession->GetOutputCount();

	std::cout << "Encoder inputs num : " << encoder_num_inputs << " outputs num : " << encoder_num_outputs << std::endl;
	std::cout << "Decoder inputs num : " << decoder_num_inputs << " outputs num : " << decoder_num_outputs << std::endl;

	encoder_input_node_names.resize(encoder_num_inputs);
	encoder_output_node_names.resize(encoder_num_outputs);

	// 原有的GetInputName和GetOutputName方法已经弃用并替换成GetInputNameAllocated()与GetOutputNameAllocated()
	std::cout << "Building Encoder num_inputs and output node nams and dims ..." << std::endl;
	for (unsigned int i = 0 ; i < encoder_num_inputs ; i++)
	{
		Ort::AllocatedStringPtr input_name = EncoderSession->GetInputNameAllocated(i , allocator);
		encoder_input_node_names.push_back(input_name.get());
	}

	for (unsigned int i = 0; i < encoder_num_outputs; i++)
	{
		Ort::AllocatedStringPtr output_name = EncoderSession->GetOutputNameAllocated(i, allocator);
		encoder_output_node_names.push_back(output_name.get());

		Ort::TypeInfo type_info = EncoderSession->GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto output_shape = tensor_info.GetShape();
		encoder_output_node_dims.push_back(output_shape);
	}

	decoder_input_node_dims.resize(decoder_num_inputs);
	decoder_output_node_names.resize(decoder_num_outputs);

	std::cout << "Building Decoder num_inputs and output node nams and dims ..." << std::endl;
	for (unsigned int i = 0; i < decoder_num_inputs; i++)
	{
		Ort::AllocatedStringPtr input_name = DecoderSession->GetInputNameAllocated(i, allocator);
		decoder_input_node_names.push_back(input_name.get());
	}

	for (unsigned int i = 0; i < decoder_num_outputs; i++)
	{
		Ort::AllocatedStringPtr output_name = DecoderSession->GetOutputNameAllocated(i, allocator);
		decoder_output_node_names.push_back(output_name.get());

		Ort::TypeInfo type_info = DecoderSession->GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto output_shape = tensor_info.GetShape();
		decoder_output_node_dims.push_back(output_shape);
	}



	delete encoder_modelpath;
	delete decoder_modelpath;
}

void SAMOnnxRunner::setSegThreshold(float threshold)
{
	SegThreshold = threshold;
}

void SAMOnnxRunner::ResetInitEncoder() 
{
	InitEncoder = false;
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

