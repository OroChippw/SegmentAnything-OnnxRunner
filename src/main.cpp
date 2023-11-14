/*
    # Author : OroChippw
    # Last Change : 2023.11.14
*/
#include <map>
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "SAMOnnxRunner.h"
#include "interactive.h"


int main(int argc , char* argv[])
{
    std::cout << "[INFO] Argc num : " << argc << std::endl;
    /* <------ CONFIG START ------> */
    // Define a vector to store parameter names and parameter values
    std::map<std::string , std::string> arguments;
    for (unsigned int i = 1 ; i < argc ; i++)
    {
        std::string arg = argv[i];
        if (arg.substr(0 , 2) == "--")
        {
            std::string param_name = arg.substr(2);
            std::string param_value = (i + 1 < argc) ? argv[i + 1] : "";
            arguments[param_name] = param_value;
            i++;
        }
    }

    for (const auto& arg : arguments)
    {
        std::cout << "[INFO] Parameter Name : " << arg.first << " , Parameter Value : " << arg.second << std::endl;
    }

    std::string encoder_model_path , decoder_model_path;
    std::string image_path , save_dir;
    bool USE_BOXINFO = true , USE_DEMO = true , USE_SINGLEMASK = false;
    double threshold = 0.9;

    for (const auto& arg : arguments)
    {
        if (arg.first == "encoder_model_path")
        {
            encoder_model_path = arg.second;
        } else if (arg.first == "decoder_model_path")
        {
            decoder_model_path = arg.second;
        } else if (arg.first == "image_path")
        {
            image_path = arg.second;
        } else if (arg.first == "save_dir")
        {
            save_dir = arg.second;
        } else if (arg.first == "use_demo")
        {
            std::istringstream(arg.second) >> std::boolalpha >> USE_DEMO;
            if (USE_DEMO) {std::cout << "[INFO] Prepare to run SAM with graphical interface demo" << std::endl;}
        } else if (arg.first == "use_boxinfo")
        {
            std::istringstream(arg.second) >> std::boolalpha >> USE_BOXINFO;
            if (USE_BOXINFO) {std::cout << "[INFO] Receive object bounding box prompt information to assist segmentation" << std::endl;}
        } else if (arg.first == "use_singlemask")
        {
            std::istringstream(arg.second) >> std::boolalpha >> USE_SINGLEMASK;
            if (USE_SINGLEMASK) {std::cout << "[INFO] The segmentation effect with the highest confidence will be output" << std::endl;}
        } else if (arg.first == "threshold")
        {
            threshold = std::stof(arg.second);
            std::cout << "[INFO] Set threshold to " << threshold << std::endl;
        }
    }

    if (encoder_model_path.empty() || decoder_model_path.empty() || image_path.empty()) {
        throw std::runtime_error("[ERROR] Model path (--encoder_model_path/--decoder_model_path) \
            or Image path (--image_path) not provided.");
    }

    if (save_dir.empty())
    {
        std::string folder_path = "../output";
        try {
            if (!std::filesystem::exists(folder_path))
            {
                std::filesystem::create_directory(folder_path);
                std::cout << "[INFO] No save folder provided, create default folder at " << folder_path << std::endl;
            } else 
            {
                std::cout << "[INFO] No save folder provided, result will save at " << folder_path << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e)
        {
            std::cerr << "[ERROR] Error creating or checking folder: " << e.what() << std::endl;
        }
    }

    unsigned int box_top_left_x , box_top_left_y , box_bot_right_x , box_bot_right_y;
    if (USE_BOXINFO)
    {
        box_top_left_x = 1333;
        box_top_left_y = 815;
        box_bot_right_x = 2048;
        box_bot_right_y = 1550; 
    }

    /* <------ CONFIG END ------> */

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

    BoxInfo boxinfo(box_top_left_x , box_top_left_y , box_bot_right_x , box_bot_right_y);
    cv::Mat srcImage = cv::imread(image_path, -1);
    cv::Mat outImage = srcImage.clone();
    if (USE_DEMO)
    {
        std::cout << "[WELCOME] Segment Anything Onnx Runner Demo" << std::endl;
        auto windowName = "Segment Anything Onnx Runner Demo";
        cv::namedWindow(windowName, 0);
        cv::setMouseCallback(
            windowName , GetClick_handler , reinterpret_cast<void*>(&clickinfo_test)
        );
        bool RunnerWork = true;
        while (RunnerWork)
        {
            if (clickinfo_test.pt.x > 0 && clickinfo_test.pt.y > 0)
            {
                auto maskinfo = Segmentator.InferenceSingleImage(cfg, srcImage, clickinfo_test, boxinfo);
                unsigned int index = 0;

                // apply mask to image
                outImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
                for (int i = 0; i < srcImage.rows; i++) {
                    for (int j = 0; j < srcImage.cols; j++) {
                        if (cfg.UseSingleMask)
                        {
                            index = 0;
                        }
                        auto bFront = maskinfo[index].mask.at<uchar>(i, j) > 0;
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
        bool RunnerWork = true;
        ClickInfo clickinfo;
        BoxInfo boxinfo;

        while (RunnerWork)
        {
            std::cout << "[PROMPT] Please enter the prompt of the click point (0/1 , x , y) : ";
            std::cin >> clickinfo.positive >> clickinfo.pt.x >> clickinfo.pt.y;

            if (cfg.UseBoxInfo)
            {
                std::cout << "[PROMPT] Please enter the prompt of the box info (x1 , y1 , x2 , y2) : ";
                std::cin >> boxinfo.left_top.x >> boxinfo.left_top.y >> boxinfo.right_bot.x >> boxinfo.right_bot.y;
            }

            auto time_start = std::chrono::high_resolution_clock::now();
            Segmentator.InferenceSingleImage(cfg, srcImage, clickinfo, boxinfo);
            auto time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = time_end - time_start;
            std::cout << "Segmentator InferenceSingleImage Cost time : " << diff.count() << "s" << std::endl;

            std::cout << "[INFO] Whether to proceed to the next round of segmentation (0/1) : ";
            std::cin >> RunnerWork; 
        }

        Segmentator.ResetInitEncoder();
    }

    return EXIT_SUCCESS;
}
    


