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
    // ClickInfo* clickinfo = reinterpret_cast<ClickInfo*>(data);
    MouseParams* mouseparams = reinterpret_cast<MouseParams*>(data);

    if (event == cv::EVENT_LBUTTONDOWN && (flags & cv::EVENT_FLAG_SHIFTKEY))
    {
        (*mouseparams).drawing = true;
        (*mouseparams).start = cv::Point(x , y);
        (*mouseparams).boxinfo.left_top = cv::Point(x , y);
    } else if (event == cv::EVENT_LBUTTONUP && (flags & cv::EVENT_FLAG_SHIFTKEY))
    {
        (*mouseparams).drawing = false;
        (*mouseparams).end = cv::Point(x , y);
        (*mouseparams).boxinfo.right_bot = cv::Point(x , y);

        std::cout << "[PROMPT] Rectangular Coordinates: " << mouseparams->start << " - " << mouseparams->end << std::endl;
    } else if (event == cv::EVENT_LBUTTONDOWN)
    {
        (*mouseparams).leftPressed = true;
        (*mouseparams).clickinfo.pt.x = x;
        (*mouseparams).clickinfo.pt.y = y;
        (*mouseparams).clickinfo.positive = 1;

        std::cout << "[PROMPT] Left Button Pressed. Coordinates: (" << x << " , " << y << ")" << std::endl;
    } else if (event == cv::EVENT_RBUTTONDOWN)
    {
        (*mouseparams).rightPressed = true;
        (*mouseparams).clickinfo.pt.x = x;
        (*mouseparams).clickinfo.pt.y = y;
        (*mouseparams).clickinfo.positive = 0;

        std::cout << "[PROMPT] Right Button Pressed. Coordinates: (" << x << " , " << y << ")" << std::endl;
    } else if (event == cv::EVENT_LBUTTONUP)
    {
        (*mouseparams).leftPressed = false;
    } else if (event == cv::EVENT_RBUTTONUP)
    {
        (*mouseparams).rightPressed = false;
    }

    if (mouseparams->drawing && (flags & cv::EVENT_FLAG_SHIFTKEY)) 
    {
        (*mouseparams).end = cv::Point(x, y);

        // 在拖拽过程中绘制矩形，可视化拖拽效果
        cv::Mat tempImage = mouseparams->image.clone();
        cv::rectangle(tempImage, mouseparams->start, mouseparams->end, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Segment Anything Onnx Runner Demo", tempImage);

    }

}
