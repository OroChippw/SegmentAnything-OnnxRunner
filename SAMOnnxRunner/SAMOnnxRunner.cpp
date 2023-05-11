﻿#pragma once

#include <iostream>
#include<Windows.h>
#include <chrono>
#include <random>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "SAMOnnxRunner.h"
#include "transform.h"
#include "utils.h"



cv::Mat SAMOnnxRunner::Image_PreProcess(cv::Mat srcImage)
{
	std::cout << "PreProcess Image ..." << std::endl;
	cv::Mat rgbImage;
	cv::cvtColor(srcImage, rgbImage, cv::COLOR_BGR2RGB);
	cv::Mat resizeImage = ResizeLongestSide_apply_image(srcImage, EncoderInputSize);
	// Normalization
	//resizeImage.convertTo(resizeImage, CV_32FC3, 1.0f / 255.0f, 0.f);

	int pad_h = EncoderInputSize - resizeImage.rows;
	int pad_w = EncoderInputSize - resizeImage.cols;

	cv::Mat paddingImage;
	cv::copyMakeBorder(resizeImage, paddingImage, 0, pad_h, 0, pad_w, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	std::cout << "paddingImage width : " << paddingImage.cols << ", paddingImage height : " << paddingImage.rows << std::endl;
	return paddingImage;
}

bool SAMOnnxRunner::Encoder_BuildEmbedding(const cv::Mat& Image)
{
	std::vector<uint8_t> inputTensorValues(EncoderInputShape[0] * EncoderInputShape[1] * EncoderInputShape[2] *
		EncoderInputShape[3]);

	if (Image.size() != cv::Size(EncoderInputShape[3], EncoderInputShape[2])) {
		std::cerr << "Image size not match" << std::endl;
		std::cout << "Image width : " << Image.cols << " Image height : " << Image.rows << std::endl;

		return false;
	}
	if (Image.channels() != 3) {
		std::cerr << "Input is not a 3-channel image" << std::endl;
		return false;
	}

	std::cout << "Encoder_BuildEmbedding Start ..." << std::endl;

	for (int i = 0; i < EncoderInputShape[2]; i++) {
		for (int j = 0; j < EncoderInputShape[3]; j++) {
			inputTensorValues[i * EncoderInputShape[3] + j] = Image.at<cv::Vec3b>(i, j)[2];
			inputTensorValues[EncoderInputShape[2] * EncoderInputShape[3] + i * EncoderInputShape[3] + j] =
				Image.at<cv::Vec3b>(i, j)[1];
			inputTensorValues[2 * EncoderInputShape[2] * EncoderInputShape[3] + i * EncoderInputShape[3] + j] =
				Image.at<cv::Vec3b>(i, j)[0];
		}
	}

	auto inputTensor = Ort::Value::CreateTensor<uint8_t>(
		memory_info_handler, inputTensorValues.data(), inputTensorValues.size(), EncoderInputShape.data(),
		EncoderInputShape.size());

	image_embedding = std::vector<float>(EncoderOutputShape[0] * EncoderOutputShape[1] * EncoderOutputShape[2] * EncoderOutputShape[3]);

	auto outputTensorPre = Ort::Value::CreateTensor<float>(
		memory_info_handler, image_embedding.data(), image_embedding.size(),
		EncoderOutputShape.data(), EncoderOutputShape.size());
	assert(outputTensorPre.IsTensor() && outputTensorPre.HasValue());

	const char* inputNamesPre[] = { "input" }, * outputNamesPre[] = { "output" };

	Ort::RunOptions run_options;
	EncoderSession->Run(run_options, inputNamesPre, &inputTensor, 1, outputNamesPre, &outputTensorPre,
		1);
	std::cout << "Encoder_BuildEmbedding Finish ..." << std::endl;

	return true;
}


std::vector<MatInfo> SAMOnnxRunner::Decoder_Inference(cv::Mat srcImage , ClickInfo clickinfo)
{
	ClickInfo applyCoords = ResizeLongestSide_apply_coord(srcImage, clickinfo, EncoderInputSize);
	std::cout << "(applyCoords.pt.x) : " << (applyCoords.pt.x) << " (applyCoords.pt.y) : " << (applyCoords.pt.y) << std::endl;
	const size_t maskInputSize = 256 * 256;
	float inputPointValues[] = { (float)applyCoords.pt.x, (float)applyCoords.pt.y }, inputLabelValues[] = { applyCoords.positive },
		maskInputValues[maskInputSize], hasMaskValues[] = { 0 },
		orig_im_size_values[] = { (float)srcImage.rows, (float)srcImage.cols};
	memset(maskInputValues, 0, sizeof(maskInputValues));

	int numPoints = 1;
	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 }, pointLabelsShape = { 1, numPoints },
		maskInputShape = { 1, 1, 256, 256 }, hasMaskInputShape = { 1 },
		origImSizeShape = { 2 };

	std::vector<Ort::Value> inputTensorsSam;
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, (float*)image_embedding.data(), image_embedding.size(),
		EncoderOutputShape.data(), EncoderOutputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, inputPointValues, 2, inputPointShape.data(), inputPointShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, inputLabelValues, 1, pointLabelsShape.data(), pointLabelsShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

	Ort::RunOptions runOptionsSam;
	auto DecoderOutputTensors = DecoderSession->Run(runOptionsSam, DecoderInputNames, inputTensorsSam.data(),
		inputTensorsSam.size(), DecoderOutputNames, 3);

	auto masks = DecoderOutputTensors[0].GetTensorMutableData<float>();
	auto iou_predictions = DecoderOutputTensors[1].GetTensorMutableData<float>();
	auto low_res_masks = DecoderOutputTensors[2].GetTensorMutableData<float>();


	Ort::Value& masks_ = DecoderOutputTensors[0];
	Ort::Value& iou_predictions_ = DecoderOutputTensors[1];
	Ort::Value& low_res_masks_ = DecoderOutputTensors[2];

	auto mask_dims = masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto iou_pred_dims = iou_predictions_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto low_res_dims = low_res_masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();

	const unsigned int Resizemasks_batch = mask_dims.at(0);
	const unsigned int Resizemasks_nums = mask_dims.at(1);
	const unsigned int Resizemasks_width = mask_dims.at(2);
	const unsigned int Resizemasks_height = mask_dims.at(3);

	std::cout << "Resizemasks_batch : " << Resizemasks_batch << " Resizemasks_nums : " << Resizemasks_nums \
		<< " Resizemasks_width : " << Resizemasks_width << " Resizemasks_height : " << Resizemasks_height << std::endl;

	std::cout << "Gemmiou_predictions_dim_0 : " << iou_pred_dims.at(0) << " Generate mask num : " << iou_pred_dims.at(1) << std::endl;

	std::cout << "Reshapelow_res_masks_dim_0 : " << low_res_dims.at(0) << " Reshapelow_res_masks_dim_1 : " << low_res_dims.at(1) << std::endl;
	std::cout << "Reshapelow_res_masks_dim_2 : " << low_res_dims.at(2) << " Reshapelow_res_masks_dim_3 : " << low_res_dims.at(3) << std::endl;


	std::vector<MatInfo> masks_list;
	for (unsigned int index = 0 ;  index < Resizemasks_nums ; index++)
	{
		cv::Mat mask(srcImage.rows, srcImage.cols, CV_8UC1);
		for (unsigned int i = 0; i < mask.rows; i++) 
		{
			for (unsigned int j = 0; j < mask.cols; j++)
			{
				mask.at<uchar>(i, j) = masks[i * mask.cols + j + index * mask.rows * mask.cols] > 0 ? 255 : 0;
			}
		}
		MatInfo mat_info;
		mat_info.mask = mask;
		mat_info.iou_pred = *(iou_predictions++);
		masks_list.emplace_back(mat_info);
	}
	return masks_list;

}

void SAMOnnxRunner::InferenceSingleImage(Configuration cfg , const cv::Mat& srcImage , ClickInfo clickInfo)
{
	if (srcImage.empty())
	{
		return;
	}
	std::cout << "Image info : srcImage width : " << srcImage.cols << " srcImage height :  " << srcImage.rows << std::endl;
	cv::Mat rgbImage = Image_PreProcess(srcImage);
	if (!InitEncoderEmbedding)
	{
		std::cout << "InitEncoder is false , Preprocess before encoder image embedding ... " << std::endl;
		auto encoder_input_tensors = Encoder_BuildEmbedding(rgbImage);
		InitEncoderEmbedding = true;
	}

	auto result = Decoder_Inference(srcImage , clickInfo);
	if (result.empty())
	{
		std::cout << "No result !" << std::endl;
		return;
	}

	std::cout << "=> Generate result mask size : " << result.size() << std::endl;

	for (unsigned int i = 0; i < result.size(); i++)
	{
		std::string save_path = cfg.SaveDir + "/result_" + std::to_string(i) +".png";
		cv::imwrite(save_path, result[i].mask);
		std::cout << "=> Result save as " << save_path << " Iou prediction is " << result[i].iou_pred << std::endl;

	}

}

void SAMOnnxRunner::InitOrtEnv(Configuration cfg) throw (std::runtime_error)
{
	// 初始化OnnxRuntime运行环境
	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SegmentAnythingModel");
	session_options = Ort::SessionOptions();
	session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
	// 设置图像优化级别
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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
	EncoderSession = std::make_unique<Ort::Session>(env, encoder_modelpath, session_options);
	if (EncoderSession->GetInputCount() != 1 || EncoderSession->GetOutputCount() != 1) {
		std::cerr << "Preprocessing model not loaded (invalid input/output count)" << std::endl;
		return;
	}
	std::cout << "Building SegmentAnything Model Decoder ... " << std::endl;
	DecoderSession = std::make_unique<Ort::Session>(env, decoder_modelpath, session_options);
	if (DecoderSession->GetInputCount() != 6 || DecoderSession->GetOutputCount() != 3) {
		std::cerr << "Model not loaded (invalid input/output count)" << std::endl;
		return;
	}

	EncoderInputShape = EncoderSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	EncoderOutputShape = EncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	if (EncoderInputShape.size() != 4 || EncoderOutputShape.size() != 4) {
		std::cerr << "Preprocessing model not loaded (invalid shape)" << std::endl;
		return;
	}
	std::cout << "=> Build EncoderSession and DecoderSession successfully." << std::endl;
	InitModelSession = true;

	delete encoder_modelpath;
	delete decoder_modelpath;
}

void SAMOnnxRunner::setSegThreshold(float threshold)
{
	SegThreshold = threshold;
}

void SAMOnnxRunner::ResetInitEncoder() 
{
	InitEncoderEmbedding = false;
}

SAMOnnxRunner::SAMOnnxRunner(unsigned int threads) :
	num_threads(threads)
{
}


