// SAMOnnxRunner.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include "SAMOnnxRunner.h"

int main()
{
    
    std::string encoder_model_path = "E:/OroChiLab/segment-anything-main/model_weights/withEncoder/vit_b/encoder.onnx";
    std::string decoder_model_path = "E:/OroChiLab/segment-anything-main/model_weights/withEncoder/vit_b/decoder.onnx";
    std::string image_path = "E:/OroChiLab/Data/NailsJpgfile/images/1_1-2.jpg";
    std::string save_dir = "E:/OroChiLab/Data/NailsJpgfile/images/savedir/";
    double threshold = 0.95;
    
    Configuration cfg;
    cfg.EncoderModelPath = encoder_model_path;
    cfg.DecoderModelPath = decoder_model_path;
    cfg.SaveDir = save_dir;
    cfg.SegThreshold = threshold;

    // Init Onnxruntime Env
    SAMOnnxRunner Segmentator;
    Segmentator.InitOrtEnv(cfg);

    ClickInfo clickinfo;
    clickinfo.positive = true;
    clickinfo.pt = cv::Point(1156,550);
    cv::Mat srcImage = cv::imread(image_path);

    Segmentator.InferenceSingleImage(cfg , srcImage , clickinfo);

    Segmentator.ResetInitEncoder();

    std::cout << "Hello World!\n";
}


