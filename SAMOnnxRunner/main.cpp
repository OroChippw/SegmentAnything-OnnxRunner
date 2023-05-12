// SAMOnnxRunner.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "SAMOnnxRunner.h"
#include "interactive.h"



int main()
{
    bool USE_DEMO = false;
    bool USE_SINGLEMASK = false;
    bool USE_BOXINFO = true;
    std::string encoder_model_path = "E:\\OroChiLab\\segment-anything-main\\models\\raw\\sam_preprocess.onnx";
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

    std::string save_dir = "E:/OroChiLab/Data/NailsJpgfile/images/savedir/";
    double threshold = 0.9;


    Configuration cfg;
    cfg.EncoderModelPath = encoder_model_path;
    cfg.DecoderModelPath = decoder_model_path;
    cfg.SaveDir = save_dir;
    cfg.SegThreshold = threshold;
    cfg.UseSingleMask = USE_SINGLEMASK;
    cfg.UseBoxInfo = USE_BOXINFO;

    // Init Onnxruntime Env
    SAMOnnxRunner Segmentator(std::thread::hardware_concurrency());
    Segmentator.InitOrtEnv(cfg);

    ClickInfo clickinfo_test;
    BoxInfo boxinfo_test;

    BoxInfo boxinfo(773, 187, 1465, 896);
    cv::Mat srcImage = cv::imread(image_path, -1);
    cv::Mat outImage = srcImage.clone();
    if (USE_DEMO)
    {
        auto windowName = "Segment Anything Onnx Runner Demo";
        cv::namedWindow(windowName, 0);
        cv::setMouseCallback(
            windowName,
            GetClick_handler,
            reinterpret_cast<void*>(&clickinfo_test)
        );
        bool RunnerWork = true;
        while (RunnerWork)
        {
            /*std::cout << clickinfo_test.pt.x << " " << clickinfo_test.pt.y << std::endl;*/
            //if (cfg.UseBoxInfo)
            //{
            //    std::cout << boxinfo_test.left_top.x << " " << boxinfo_test.left_top.y << std::endl;
            //    std::cout << boxinfo_test.right_bot.x << " " << boxinfo_test.right_bot.y << std::endl;

            //}
            if (clickinfo_test.pt.x > 0 && clickinfo_test.pt.y > 0)
            {
                auto maskinfo = Segmentator.InferenceSingleImage(cfg, srcImage, clickinfo_test, boxinfo);

                // apply mask to image
                outImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
                for (int i = 0; i < srcImage.rows; i++) {
                    for (int j = 0; j < srcImage.cols; j++) {
                        auto bFront = maskinfo[1].mask.at<uchar>(i, j) > 0;
                        float factor = bFront ? 1.0 : 0.5;
                        outImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(i, j) * factor;
                    }
                }
            }
            clickinfo_test.pt.x = 0;
            clickinfo_test.pt.y = 0;
            cv::imshow(windowName, outImage);
            int key = cv::waitKeyEx(100);
            switch (key) {
            case 27:
            case 'q': {
                RunnerWork = false;
            } break;
            case 'c': {
                outImage = srcImage.clone();
            } break;
            }
        }

        cv::destroyWindow(windowName);
    }
    else
    {
        ClickInfo clickinfo;
        clickinfo.positive = true;
        clickinfo.pt = cv::Point(1156, 550);

        if (cfg.UseBoxInfo)
        {
            BoxInfo boxinfo_(773, 187, 1465, 896);
        }
        auto time_start = std::chrono::high_resolution_clock::now();
        Segmentator.InferenceSingleImage(cfg, srcImage, clickinfo, boxinfo);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;
        std::cout << "Segmentator InferenceSingleImage Cost time : " << diff.count() << "s" << std::endl;

        Segmentator.ResetInitEncoder();
    }
    
    std::cout << "Hello World!\n";
}
    


