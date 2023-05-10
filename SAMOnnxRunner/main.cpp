// SAMOnnxRunner.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "SAMOnnxRunner.h"
#include "sam.h"

int main()
{
    bool USE_SINGLEMASK = false;
    std::string encoder_model_path = "E:/OroChiLab/segment-anything-main/model_weights/withEncoder/vit_b/encoder.onnx";
    std::string decoder_model_path;
    if (USE_SINGLEMASK)
    {
        decoder_model_path = "E:/OroChiLab/segment-anything-main/model_weights/sam_vit_b_singlemask.onnx";

    }
    else
    {
        decoder_model_path = "E:/OroChiLab/segment-anything-main/model_weights/withEncoder/vit_b/decoder.onnx";
    }
    std::string image_path = "E:/OroChiLab/Data/NailsJpgfile/images/1_1-2.jpg";
    //for (auto& p : {&encoder_model_path , &decoder_model_path})
    //{
    //    std::ifstream f(p);
    //    if (!f.good())
    //    {
    //        std::cerr << "Model file " << p << " is not exist" << < std::endl;
    //    }
    //}
    std::string save_dir = "E:/OroChiLab/Data/NailsJpgfile/images/savedir/";
    double threshold = 0.95;



    Configuration cfg;
    cfg.EncoderModelPath = encoder_model_path;
    cfg.DecoderModelPath = decoder_model_path;
    cfg.SaveDir = save_dir;
    cfg.SegThreshold = threshold;

    // Init Onnxruntime Env
    SAMOnnxRunner Segmentator(std::thread::hardware_concurrency());
    Segmentator.InitOrtEnv(cfg);

    ClickInfo clickinfo;
    clickinfo.positive = true;
    clickinfo.pt = cv::Point(1156, 550);
    cv::Mat srcImage = cv::imread(image_path);

    Segmentator.InferenceSingleImage(cfg, srcImage, clickinfo);

    Segmentator.ResetInitEncoder();

    std::cout << "Hello World!\n";
}
    


