/*
    # Author : OroChippw
    # Last Change : 2023.11.20
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
    for (int i = 1 ; i < argc ; i++)
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
    bool USE_BOXINFO = false , USE_DEMO = true , USE_SINGLEMASK = false;
    bool KEEP_BOXINFO = true;
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
        std::cout << "[ERROR] Model path (--encoder_model_path/--decoder_model_path) or Image path (--image_path) not provided." << std::endl;
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

    /* <------ CONFIG END ------> */

    Configuration cfg;
    cfg.EncoderModelPath = encoder_model_path;
    cfg.DecoderModelPath = decoder_model_path; 
    cfg.SaveDir = save_dir;
    cfg.SegThreshold = threshold;
    cfg.UseSingleMask = USE_SINGLEMASK;
    cfg.UseBoxInfo = USE_BOXINFO;
    cfg.KeepBoxInfo = KEEP_BOXINFO;

    // Init Onnxruntime Env
    SAMOnnxRunner Segmentator(std::thread::hardware_concurrency());
    Segmentator.InitOrtEnv(cfg);

    cv::Mat srcImage = cv::imread(image_path, -1);
    cv::Mat visualImage = srcImage.clone();
    cv::Mat maskImage;

    if (USE_DEMO)
    {
        std::cout << "[WELCOME] Segment Anything Onnx Runner Demo" << std::endl;
        auto windowName = "Segment Anything Onnx Runner Demo";

        MouseParams mouseparams;

        mouseparams.image = visualImage;
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName , srcImage.cols / 2 , srcImage.rows / 2);

        cv::setMouseCallback(
            windowName , GetClick_handler , reinterpret_cast<void*>(&mouseparams)
        );
        
        bool RunnerWork = true;
        while (RunnerWork)
        {
            std::vector<MatInfo> maskinfo;
            // 当产生有效点击时才进行推理
            if ((mouseparams.clickinfo.pt.x > 0) && (mouseparams.clickinfo.pt.y > 0))
            {
                maskinfo = Segmentator.InferenceSingleImage(cfg, srcImage, mouseparams.clickinfo , mouseparams.boxinfo);
                unsigned int index = 0;
                // Apply mask to image
                visualImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
                for (int i = 0; i < srcImage.rows; i++) {
                    for (int j = 0; j < srcImage.cols; j++) {
                        double factor = maskinfo[index].mask.at<uchar>(i, j) > 0 ? 1.0 : 0.4;
                        visualImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(i, j) * factor;
                    }
                }
                maskImage = maskinfo[0].mask;
            }
            mouseparams.clickinfo.pt.x = 0;
            mouseparams.clickinfo.pt.y = 0;
            if (!cfg.KeepBoxInfo)
            {
                mouseparams.boxinfo.left_top = cv::Point(0 , 0);
                mouseparams.boxinfo.right_bot = cv::Point(srcImage.cols , srcImage.rows);
            }

            cv::imshow(windowName, visualImage);
 
            int key = cv::waitKeyEx(100);
            switch (key) {
            case 27:
            case 'q': { // Quit Segment Anything Onnx Runner Demo
                RunnerWork = false;
            } break;
            case 'c': { 
                // Continue , Use the mask output from the previous run. 
                if (!(maskImage.empty())) 
                {
                    std::cout << "[INFO] The maskImage is not empty, and the mask with the highest confidence is used as the mask_input of the decoder." << std::endl;
                    cfg.HasMaskInput = true;
                } else 
                {
                    std::cout << "[WARNINGS] The maskImage is empty, and there is no available mask as the mask_input of the decoder." << std::endl;
                }
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
    


