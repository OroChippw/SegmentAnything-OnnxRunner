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
	cv::Mat resizeImage = ResizeLongestSide_apply_image(Image , EncoderInputSize);
	
	 // Normalization
	resizeImage.convertTo(resizeImage, CV_32FC3, 1.0f / 255.0f, 0.f);
	
	int pad_h = EncoderInputSize - resizeImage.rows;
	int pad_w = EncoderInputSize - resizeImage.cols;

	cv::Mat paddingImage;
	cv::copyMakeBorder(resizeImage , paddingImage , 0 ,pad_h , 0 , pad_w , cv::BorderTypes::BORDER_CONSTANT , cv::Scalar(0,0,0));
	
	// 构造输入Encoder的tensor
	const unsigned int target_height = encoder_input_node_dims.at(0).at(2);
	const unsigned int target_width = encoder_input_node_dims.at(0).at(3);
	const unsigned int target_channel = encoder_input_node_dims.at(0).at(1);
	/*std::cout << "target_channel : " << target_channel << std::endl;
	std::cout << "target_width : " << target_width << std::endl;
	std::cout << "target_height : " << target_height << std::endl;*/

	const unsigned int target_tensor_size = target_channel * target_height * target_width;

	std::cout << "Resize input value handler to " << target_tensor_size << std::endl;
	input_bgr_value_handler.resize(target_tensor_size);

	int flag = 1;
	if (flag == 1)
	{
		// C * H * W
		std::vector<cv::Mat> mat_channels;
		cv::split(paddingImage, mat_channels);
		if (target_channel != channels)
		{
			throw std::runtime_error("Runtime Error Channel mismatch.");
		}
		for (unsigned int i = 0; i < channels; i++)
		{
			std::memcpy(input_bgr_value_handler.data() + i * (target_height * target_width),
				mat_channels.at(i).data, target_height * target_width * sizeof(float));
		}
	}
	else
	{
		// H * W * C
		std::memcpy(input_bgr_value_handler.data(), paddingImage.data, target_tensor_size * sizeof(float));
	}
	
	std::cout << "=> Create Encoder input tensor ..." << std::endl;

	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info_handler, input_bgr_value_handler.data(),
		target_tensor_size, encoder_input_node_dims.at(0).data(),
		encoder_input_node_dims.at(0).size());

	assert(input_tensor.IsTensor());

	return input_tensor;
}


std::vector<Ort::Value> SAMOnnxRunner::Encoder_BuildEmbedding(Ort::Value* input_tensors) throw (std::runtime_error)
{
	std::cout << "=> Encoder build image embedding start ... " << std::endl;
	auto start_time = std::chrono::steady_clock::now();
	
	std::vector<Ort::Value> output_tensors = EncoderSession->Run(
		Ort::RunOptions{nullptr} , encoder_input_node_names.data() , 
		input_tensors, encoder_num_inputs, encoder_output_node_names.data(), 
		encoder_num_outputs
	);
	
	auto end_time = std::chrono::steady_clock::now();
	std::cout << "Encoder build image embedding finish ... " << std::endl;
	std::cout << "Encoder build embedding cost time : " << \
		std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
	
	return output_tensors;
}

std::vector<Ort::Value> SAMOnnxRunner::Decoder_PreProcess(cv::Mat Image, ClickInfo clickinfo) throw (std::runtime_error)
{
	ClickInfo applyCoords = ResizeLongestSide_apply_coord(Image, clickinfo, EncoderInputSize);
	std::cout << "(applyCoords.pt.x) : " << (applyCoords.pt.x) << " (applyCoords.pt.y) : " << (applyCoords.pt.y) << std::endl;

	/*
	* TODO 解码器几个输入tensor的大小
	* 验证是否可以使用通用的memory_info_handler
	* Reference : https://gist.github.com/tempdeltavalue/2ddf7d3195f336d2a1dd7d5e71e28224
	*/
	std::cout << "=> Create Decoder input tensor ..." << std::endl;

	decoder_input_node_dims = {
		{1 , 256 , 64 , 64} , // image_embeddings
		{1 , 1 , 2} , // point_coords
		{1 , 1} , // point_labels
		{1 , 1 , 256 , 256} , // mask_inputs
		{1} , // has_mask_input
		{2} , // orig_im_size
	};

	std::vector<Ort::Value> output_tensors;

	output_tensors.push_back(std::move(image_embedding[0]));

	std::cout << "Building decoder input tensors [point coords] ..." << std::endl;
	auto memory_info_point_coords = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::array<float , 2> point_coords = {float(applyCoords.pt.x) , float(applyCoords.pt.y)};
	
	auto point_coords_tensors = Ort::Value::CreateTensor<float>(
		memory_info_point_coords, point_coords.data() , point_coords.size() ,
		decoder_input_node_dims.at(1).data() , decoder_input_node_dims.at(1).size()
	);
	assert(point_coords_tensors.IsTensor());
	output_tensors.push_back(std::move(point_coords_tensors));


	std::cout << "Building decoder input tensors [point labels] ..." << std::endl;
	auto memory_info_point_labels = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::array<float , 1> point_labels = { float(int(applyCoords.positive)) };
	auto point_labels_tensors = Ort::Value::CreateTensor<float>(
		memory_info_point_labels, point_labels.data(), point_labels.size(),
		decoder_input_node_dims.at(2).data(), decoder_input_node_dims.at(2).size()
	);
	assert(point_labels_tensors.IsTensor());
	output_tensors.push_back(std::move(point_labels_tensors));


	std::cout << "Building decoder input tensors [mask inputs] ..." << std::endl;
	auto memory_info_mask_inputs = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::array<float, 256 * 256> mask_inputs{};
	auto mask_inputs_tensors = Ort::Value::CreateTensor<float>(
		memory_info_mask_inputs, mask_inputs.data(), mask_inputs.size(),
		decoder_input_node_dims.at(3).data(), decoder_input_node_dims.at(3).size()
	);
	assert(mask_inputs_tensors.IsTensor());
	output_tensors.push_back(std::move(mask_inputs_tensors));


	std::cout << "Building decoder input tensors [has_mask_input] ..." << std::endl;
	auto memory_info_has_mask_input = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	std::array<float, 1> has_mask_input = {0};
	auto has_mask_inputs_tensors = Ort::Value::CreateTensor<float>(
		memory_info_has_mask_input, has_mask_input.data(), has_mask_input.size(),
		decoder_input_node_dims.at(4).data(), decoder_input_node_dims.at(4).size()
	);
	assert(has_mask_inputs_tensors.IsTensor());
	output_tensors.push_back(std::move(has_mask_inputs_tensors));
	
	std::cout << "Building decoder input tensors [orig_im_size] ..." << std::endl;
	auto memory_info_orig_im_size = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::vector<float>  orig_im_size = {float(Image.cols) , float(Image.rows)};

	auto orig_im_size_tensors = Ort::Value::CreateTensor<float>(
		memory_info_orig_im_size, orig_im_size.data(), orig_im_size.size(),
		decoder_input_node_dims.at(5).data(), decoder_input_node_dims.at(5).size()
	);

	assert(orig_im_size_tensors.IsTensor());
	output_tensors.push_back(std::move(orig_im_size_tensors));

	return output_tensors;

}


std::vector<Ort::Value> SAMOnnxRunner::Decoder_Inference(std::vector<Ort::Value>* input_tensors) throw (std::runtime_error)
{
	std::cout << "=> Decoder inference start ... " << std::endl;
	auto start_time = std::chrono::steady_clock::now();

	std::vector<Ort::Value> output_tensors = DecoderSession->Run(
		Ort::RunOptions{nullptr} , decoder_input_node_names.data() , 
		input_tensors->data(), decoder_num_inputs, decoder_output_node_names.data(),
		decoder_num_outputs
	);
	
	auto end_time = std::chrono::steady_clock::now();

	std::cout << "Decoder inference finish ... " << std::endl;
	std::cout << "Decoder Inference cost time : " << \
		std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;
	
	return output_tensors;
}

void SAMOnnxRunner::Encoder_PostProcess()
{
	// 编码器暂时不需要进行后处理，直接将embedding输入解码器配合点击事件即可
}

void SAMOnnxRunner::Decoder_PostProcess()
{

}

void SAMOnnxRunner::InferenceSingleImage(Configuration cfg , cv::Mat srcImage , ClickInfo clickInfo)
{
	std::cout << "Image info : srcImage width : " << srcImage.cols << " srcImage height :  " << srcImage.rows << std::endl;
	if (srcImage.empty())
	{
		return;
	}
	cv::Mat rgbImage = Image_PreProcess(srcImage);
	if (!InitEncoder)
	{
		std::cout << "InitEncoder is false , Preprocess before encoder image embedding ... " << std::endl;
		auto encoder_input_tensors = Encoder_PreProcess(rgbImage);
		image_embedding = std::move(Encoder_BuildEmbedding(&encoder_input_tensors));
		InitEncoder = true;
	}
	
	auto decoder_input_tensors = std::move(Decoder_PreProcess(srcImage, clickInfo));
	auto decoder_output_tensors = std::move(Decoder_Inference(&decoder_input_tensors));

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
	std::cout << "=> Using Onnxruntime C++ API." << std::endl;
	std::cout << "Building SegmentAnything Model Encoder ... " << std::endl;
	EncoderSession = new Ort::Session(env , encoder_modelpath , session_options);
	std::cout << "Building SegmentAnything Model Decoder ... " << std::endl;
	DecoderSession = new Ort::Session(env, decoder_modelpath, session_options);

	allocator = Ort::AllocatorWithDefaultOptions();

	std::cout << "=> Build EncoderSession and DecoderSession successfully." << std::endl;

	// Get Session input and output info
	encoder_num_inputs = EncoderSession->GetInputCount(); 	// Encoder
	encoder_num_outputs = EncoderSession->GetOutputCount();

	std::cout << "Encoder inputs num : " << encoder_num_inputs << " outputs num : " << encoder_num_outputs << std::endl;

	encoder_input_node_names.resize(encoder_num_inputs);
	encoder_output_node_names.resize(encoder_num_outputs);


	// 原有的GetInputName和GetOutputName方法已经弃用并替换成GetInputNameAllocated()与GetOutputNameAllocated()

	// OnnxRuntime1.13.x版本以后，GetInputName类似的函数存在泄露的问题
	// 新版api的返回值是一个unique_ptr指针，这就意味着他使用一次时候就被销毁了,用向量将GetInputNameAllocated结果保存下来

	std::cout << "Building Encoder num_inputs and output node nams and dims ..." << std::endl;
	for (unsigned int i = 0 ; i < encoder_num_inputs ; i++)
	{	
		Ort::AllocatorWithDefaultOptions allocator;
		Encoder_In_AllocatedStringPtr.push_back(EncoderSession->GetInputNameAllocated(i, allocator));
		encoder_input_node_names[i] = (Encoder_In_AllocatedStringPtr.at(i).get());
		Ort::TypeInfo input_type_info = EncoderSession->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		encoder_input_node_dims.push_back(input_dims);
	}

	for (unsigned int i = 0; i < encoder_num_outputs; i++)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		Encoder_Out_AllocatedStringPtr.push_back(EncoderSession->GetOutputNameAllocated(i, allocator));
		encoder_output_node_names[i] = (Encoder_Out_AllocatedStringPtr.at(i).get());

		Ort::TypeInfo type_info = EncoderSession->GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto output_shape = tensor_info.GetShape();
		encoder_output_node_dims.push_back(output_shape);
	}

	// ******************************************************* //

	decoder_num_inputs = DecoderSession->GetInputCount(); // Decoder
	decoder_num_outputs = DecoderSession->GetOutputCount();

	decoder_input_node_names.resize(decoder_num_inputs);
	decoder_output_node_names.resize(decoder_num_outputs);

	std::cout << "Decoder inputs num : " << decoder_num_inputs << " outputs num : " << decoder_num_outputs << std::endl;


	std::cout << "Building Decoder num_inputs and output node nams and dims ..." << std::endl;
	for (unsigned int i = 0; i < decoder_num_inputs; i++)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		Decoder_In_AllocatedStringPtr.push_back(DecoderSession->GetInputNameAllocated(i, allocator));
		
		decoder_input_node_names[i] = (Decoder_In_AllocatedStringPtr.at(i).get());
		Ort::TypeInfo input_type_info = DecoderSession->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		decoder_input_node_dims.push_back(input_dims);
	}

	printf("1");

	for (unsigned int i = 0; i < decoder_num_outputs; i++)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		Decoder_Out_AllocatedStringPtr.push_back(DecoderSession->GetOutputNameAllocated(i, allocator));
		decoder_output_node_names[i] = (Decoder_Out_AllocatedStringPtr.at(i).get());
		// org
		/*Ort::AllocatedStringPtr output_name = EncoderSession->GetOutputNameAllocated(i, allocator);
		encoder_output_node_names.push_back(output_name.get());*/

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

