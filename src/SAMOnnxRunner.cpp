#pragma once

#include <iostream>
#include<Windows.h>
#include <chrono>
#include <random>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>  // If running in a GPU environment, cuda can be used for acceleration.

#include "SAMOnnxRunner.h"
#include "transform.h"
#include "utils.h"



cv::Mat SAMOnnxRunner::Image_PreProcess(cv::Mat srcImage)
{
	std::cout << "[INFO] PreProcess Image ..." << std::endl;
	cv::Mat rgbImage;
	cv::cvtColor(srcImage, rgbImage, cv::COLOR_BGR2RGB);

	cv::Mat floatImage;
	rgbImage.convertTo(floatImage, CV_32FC3);
	cv::Mat pixelMean = cv::Mat::ones(cv::Size(floatImage.cols, floatImage.rows), CV_32FC3);
	cv::Mat pixelStd = cv::Mat::ones(cv::Size(floatImage.cols, floatImage.rows), CV_32FC3);
	pixelMean = cv::Scalar(123.675, 116.28, 103.53);
	pixelStd = cv::Scalar(58.395, 57.12, 57.375);
	floatImage -= pixelMean;
	floatImage /= pixelStd;

	cv::Mat resizeImage = ResizeLongestSide_apply_image(floatImage, EncoderInputSize);

	int pad_h = EncoderInputSize - resizeImage.rows;
	int pad_w = EncoderInputSize - resizeImage.cols;

	cv::Mat paddingImage;
	cv::copyMakeBorder(resizeImage, paddingImage, 0, pad_h, 0, pad_w, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	std::cout << "[INFO] paddingImage width : " << paddingImage.cols << ", paddingImage height : " << paddingImage.rows << std::endl;
	return paddingImage;
}

bool SAMOnnxRunner::Encoder_BuildEmbedding(const cv::Mat& Image)
{
	std::vector<uint8_t> inputTensorValues(EncoderInputShape[0] * EncoderInputShape[1] * EncoderInputShape[2] *
		EncoderInputShape[3]);

	if (Image.size() != cv::Size(EncoderInputShape[3], EncoderInputShape[2])) {
		std::cerr << "[WARRING] Image size not match" << std::endl;
		std::cout << "[INFO] Image width : " << Image.cols << " Image height : " << Image.rows << std::endl;

		return false;
	}
	if (Image.channels() != 3) {
		std::cerr << "Input image is not a 3-channel image" << std::endl;
		return false;
	}

	std::cout << "[INFO] Encoder BuildEmbedding Start ..." << std::endl;
	for (int i = 0; i < EncoderInputShape[2]; i++) {
		for (int j = 0; j < EncoderInputShape[3]; j++) {
			inputTensorValues[i * EncoderInputShape[3] + j] = Image.at<cv::Vec3f>(i, j)[2];
			inputTensorValues[EncoderInputShape[2] * EncoderInputShape[3] + i * EncoderInputShape[3] + j] =
				Image.at<cv::Vec3f>(i, j)[1];
			inputTensorValues[2 * EncoderInputShape[2] * EncoderInputShape[3] + i * EncoderInputShape[3] + j] =
				Image.at<cv::Vec3f>(i, j)[0];
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

	auto time_start = std::chrono::high_resolution_clock::now();
	Ort::RunOptions run_options;
	EncoderSession->Run(run_options, inputNamesPre, &inputTensor, 1, outputNamesPre, &outputTensorPre,
		1);
	auto time_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = time_end - time_start;

	std::cout << "[INFO] Encoder BuildEmbedding Finish ..." << std::endl;
	std::cout << "[INFO] Encoder BuildEmbedding Cost time : " << diff.count() << "s" << std::endl;

	return true;
}


std::vector<MatInfo> SAMOnnxRunner::Decoder_Inference(Configuration cfg , cv::Mat srcImage , ClickInfo clickinfo , BoxInfo boxinfo)
{
	int numPoints = cfg.UseBoxInfo ? 3 : 1;
	std::cout << "[INFO] ResizeLongestSide apply coordinates" << std::endl;
	ClickInfo applyCoords = ResizeLongestSide_apply_coord(srcImage, clickinfo, EncoderInputSize);
	std::cout << "[INFO] (applyCoords.pt.x) : " << (applyCoords.pt.x) << " (applyCoords.pt.y) : " << (applyCoords.pt.y) << std::endl;
	float inputPointValues[] = { (float)applyCoords.pt.x, (float)applyCoords.pt.y };
	float inputLabelValues[] = {static_cast<float>(applyCoords.positive)};


	BoxInfo applyBoxs = ResizeLongestSide_apply_box(srcImage, boxinfo, EncoderInputSize);
	std::cout << "[INFO] (applyBoxs.left_top.x) : " << (applyBoxs.left_top.x) << " (applyBoxs.left_top.y) : " << (applyBoxs.left_top.y) << std::endl;
	std::cout << "[INFO] (applyBoxs.right_bot.x) : " << (applyBoxs.right_bot.x) << " (applyBoxs.right_bot.y) : " << (applyBoxs.right_bot.y) << std::endl;

	float inputPointsValues[] = {(float)applyCoords.pt.x, (float)applyCoords.pt.y ,\
								  (float)applyBoxs.left_top.x, (float)applyBoxs.left_top.y ,\
								  (float)applyBoxs.right_bot.x, (float)applyBoxs.right_bot.y ,
								};
	float inputLabelsValues[] = {static_cast<float>(applyCoords.positive) , (float)2 , (float)3 };


	const size_t maskInputSize = 256 * 256;
	float maskInputValues[maskInputSize], hasMaskValues[] = { 0 },
		orig_im_size_values[] = { (float)srcImage.rows, (float)srcImage.cols};
	memset(maskInputValues, 0, sizeof(maskInputValues));

	std::vector<int64_t> inputPointShape = { 1, numPoints, 2 }, pointLabelsShape = { 1, numPoints },
		maskInputShape = { 1, 1, 256, 256 }, hasMaskInputShape = { 1 },
		origImSizeShape = { 2 };

	std::vector<Ort::Value> inputTensorsSam;
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, (float*)image_embedding.data(), image_embedding.size(),
		EncoderOutputShape.data(), EncoderOutputShape.size()));
	if (cfg.UseBoxInfo)
	{
		inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
			memory_info_handler, inputPointsValues, 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
		inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
			memory_info_handler, inputLabelsValues, 1 * numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
	}
	else
	{
		inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
			memory_info_handler, inputPointValues, 2, inputPointShape.data(), inputPointShape.size()));
		inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
			memory_info_handler, inputLabelValues, 1, pointLabelsShape.data(), pointLabelsShape.size()));
	}
	
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
	inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
		memory_info_handler, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

	Ort::RunOptions runOptionsSam;

	std::cout << "[INFO] Decoder Inference Start ..." << std::endl;
	auto time_start = std::chrono::high_resolution_clock::now();
	auto DecoderOutputTensors = DecoderSession->Run(runOptionsSam, DecoderInputNames, inputTensorsSam.data(),
		inputTensorsSam.size(), DecoderOutputNames, 3);
	auto time_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = time_end - time_start;
	std::cout << "[INFO] Decoder Inference Finish ..." << std::endl;
	std::cout << "[INFO] Decoder Inference Cost time : " << diff.count() << "s" << std::endl;

	auto masks = DecoderOutputTensors[0].GetTensorMutableData<float>();
	auto iou_predictions = DecoderOutputTensors[1].GetTensorMutableData<float>();
	auto low_res_masks = DecoderOutputTensors[2].GetTensorMutableData<float>();


	Ort::Value& masks_ = DecoderOutputTensors[0];
	Ort::Value& iou_predictions_ = DecoderOutputTensors[1];
	Ort::Value& low_res_masks_ = DecoderOutputTensors[2];

	auto mask_dims = masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto iou_pred_dims = iou_predictions_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
	auto low_res_dims = low_res_masks_.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();

	const int Resizemasks_batch = static_cast<int>(mask_dims.at(0));
	const int Resizemasks_nums = static_cast<int>(mask_dims.at(1));
	const int Resizemasks_width = static_cast<int>(mask_dims.at(2));
	const int Resizemasks_height = static_cast<int>(mask_dims.at(3));

	std::cout << "[INFO] Resizemasks_batch : " << Resizemasks_batch << " Resizemasks_nums : " << Resizemasks_nums \
		<< " Resizemasks_width : " << Resizemasks_width << " Resizemasks_height : " << Resizemasks_height << std::endl;

	std::cout << "[INFO] Gemmiou_predictions_dim_0 : " << iou_pred_dims.at(0) << " Generate mask num : " << iou_pred_dims.at(1) << std::endl;

	std::cout << "[INFO] Reshapelow_res_masks_dim_0 : " << low_res_dims.at(0) << " Reshapelow_res_masks_dim_1 : " << low_res_dims.at(1) << std::endl;
	std::cout << "[INFO] Reshapelow_res_masks_dim_2 : " << low_res_dims.at(2) << " Reshapelow_res_masks_dim_3 : " << low_res_dims.at(3) << std::endl;


	std::vector<MatInfo> masks_list;
	for (int index = 0 ;  index < Resizemasks_nums ; index++)
	{
		cv::Mat mask(srcImage.rows, srcImage.cols, CV_8UC1);
		for (int i = 0; i < mask.rows; i++) 
		{
			for (int j = 0; j < mask.cols; j++)
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


std::vector<MatInfo> SAMOnnxRunner::InferenceSingleImage(Configuration cfg , const cv::Mat& srcImage , ClickInfo clickInfo , BoxInfo boxinfo)
{
	if (srcImage.empty())
	{
		throw "[ERROR] srcImage empty !";
	}
	std::cout << "[INFO] Image info : srcImage width : " << srcImage.cols << " srcImage height :  " << srcImage.rows << std::endl;
	cv::Mat rgbImage = Image_PreProcess(srcImage);

	if (!InitEncoderEmbedding)
	{
		std::cout << "[INFO] InitEncoder is false , Preprocess before encoder image embedding ... " << std::endl;
		auto encoder_input_tensors = Encoder_BuildEmbedding(rgbImage);
		InitEncoderEmbedding = true;
	}
	auto result = Decoder_Inference(cfg , srcImage , clickInfo , boxinfo);
	if (result.empty())
	{
		throw  "[ERROR] No result !" ;
	}else {
		std::sort(result.begin(), result.end(), [](const MatInfo& a, const MatInfo& b) {
        	return a.iou_pred > b.iou_pred;
		});
	}

	std::cout << "[INFO] Generate result mask size is " << result.size() << std::endl;

	unsigned int index = 0;
	for (unsigned int i = 0; i < result.size(); i++)
	{	
		if (result[i].iou_pred < cfg.SegThreshold)
		{
			std::cout << "[INFO] Result IoU prediction lower than segthreshold , only " << result[i].iou_pred << std::endl;
			continue;
		}
		std::string save_path = cfg.SaveDir + "/result_" + std::to_string(i) +".png";
		cv::imwrite(save_path, result[i].mask);
		std::cout << "[INFO] Result save as " << save_path << " Iou prediction is " << result[i].iou_pred << std::endl;
	}
	return result;
}

void SAMOnnxRunner::InitOrtEnv(Configuration cfg)
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
	std::cout << "[ENV] Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
	const wchar_t* encoder_modelpath = multi_Byte_To_Wide_Char(cfg.EncoderModelPath);
	const wchar_t* decoder_modelpath = multi_Byte_To_Wide_Char(cfg.DecoderModelPath);
#else
	const char* encoder_modelpath = cfg.EncoderModelPath;
	const char* decoder_modelpath = cfg.DecoderModelPath;
#endif // _WIN32

	// 创建Session并加载模型进内存
	std::cout << "[INFO] Using Onnxruntime C++ API." << std::endl;
	std::cout << "[INFO] Building SegmentAnything Model Encoder ... " << std::endl;
	EncoderSession = std::make_unique<Ort::Session>(env, encoder_modelpath, session_options);
	if (EncoderSession->GetInputCount() != 1 || EncoderSession->GetOutputCount() != 1) {
		std::cerr << "[INFO] Preprocessing model not loaded (invalid input/output count)" << std::endl;
		return;
	}
	std::cout << "[INFO] Building SegmentAnything Model Decoder ... " << std::endl;
	DecoderSession = std::make_unique<Ort::Session>(env, decoder_modelpath, session_options);
	if (DecoderSession->GetInputCount() != 6 || DecoderSession->GetOutputCount() != 3) {
		std::cerr << "[INFO] Model not loaded (invalid input/output count)" << std::endl;
		return;
	}

	EncoderInputShape = EncoderSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	EncoderOutputShape = EncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	if (EncoderInputShape.size() != 4 || EncoderOutputShape.size() != 4) {
		std::cerr << "[ERROR] Preprocessing model not loaded (invalid shape)" << std::endl;
		return;
	}
	std::cout << "[INFO] Build EncoderSession and DecoderSession successfully." << std::endl;
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


