#pragma once
#include <opencv2/opencv.hpp>
#include "SAMOnnxRunner.h"

struct MouseParams 
{
    cv::Mat image;

    cv::Point start;
    cv::Point end;

    ClickInfo clickinfo;
    BoxInfo boxinfo;

    bool drawing;
    bool leftPressed;
    bool rightPressed;
};

void GetClick_handler(int event, int x, int y, int flags, void* data)
{
    ClickInfo* clickinfo = reinterpret_cast<ClickInfo*>(data);
    MouseParams* mouseparams = reinterpret_cast<MouseParams*>(data);

    if (event == cv::EVENT_LBUTTONDOWN && (flags & cv::EVENT_FLAG_SHIFTKEY))
    {
        mouseparams->drawing = true;
        mouseparams->start = cv::Point(x , y);
    } else if (event == cv::EVENT_LBUTTONUP && (flags & cv::EVENT_FLAG_SHIFTKEY))
    {
        mouseparams->drawing = false;
        mouseparams->end = cv::Point(x , y);
        std::cout << "[PROMPT] Rectangular Coordinates: " << mouseparams->start << " - " << mouseparams->end << std::endl;
    } else if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouseparams->leftPressed = true;
        mouseparams->clickinfo.pt.x = x;
        mouseparams->clickinfo.pt.y = y;
        mouseparams->clickinfo.positive = 1;

        std::cout << "[PROMPT] Left Button Pressed. Coordinates: " << x << " , " << y << std::endl;
    } else if (event == cv::EVENT_RBUTTONDOWN)
    {
        mouseparams->rightPressed = true;

        mouseparams->clickinfo.pt.x = x;
        mouseparams->clickinfo.pt.y = y;
        mouseparams->clickinfo.positive = 0;

        std::cout << "[PROMPT] Right Button Pressed. Coordinates: " << x << " , " << y << std::endl;
    } else if (event == cv::EVENT_LBUTTONUP)
    {
        mouseparams->leftPressed = false;
    } else if (event == cv::EVENT_RBUTTONUP)
    {
        mouseparams->rightPressed = false;
    }

    if (mouseparams->drawing && (flags & cv::EVENT_FLAG_SHIFTKEY)) 
    {
        mouseparams->end = cv::Point(x, y);

        // 在拖拽过程中绘制矩形，可视化拖拽效果
        cv::Mat tempImage = mouseparams->image.clone();
        cv::rectangle(tempImage, mouseparams->start, mouseparams->end, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Window", tempImage);
    }

    // switch (event)
    // {
    // case cv::EVENT_LBUTTONDOWN:
    // {
    //     std::cout << "[INFO] Mouse handler trigger : EVENT_LBUTTONDOWN" << std::endl;
    //     (*clickinfo).pt.x = x;
    //     (*clickinfo).pt.y = y;
    //     (*clickinfo).positive = 1;
    //     break;
    // }
    // case cv::EVENT_RBUTTONDOWN:
    // {
    //     std::cout << "[INFO] Mouse handler trigger : EVENT_RBUTTONDOWN" << std::endl;
    //     (*clickinfo).pt.x = x;
    //     (*clickinfo).pt.y = y;
    //     (*clickinfo).positive = 0;
    //     break;
    // }
    // default:
    //     break;
    // }


}


void GetBox_handler(int event , int x , int y , int flags , void* data)
{
    BoxInfo* boxinfo = reinterpret_cast<BoxInfo*>(data);
    cv::Point left_top;
    cv::Point right_bot;
    switch (event)
    {
    case (cv::EVENT_LBUTTONDOWN):
    {
        left_top.x = x;
        left_top.y = y;
        break;
    }
    case (cv::EVENT_MOUSEMOVE && cv::EVENT_LBUTTONUP):
    {
        right_bot.x = x;
        right_bot.y = y;
        break;
    }
    /*case ():
    {
        right_bot.x;

        break;
    }*/
    default:
        break;
    }
    if (left_top.x > 0 && left_top.y > 0 && right_bot.x > 0 && right_bot.y)
    {
        boxinfo->left_top = left_top;
        boxinfo->right_bot = right_bot;
    }
}