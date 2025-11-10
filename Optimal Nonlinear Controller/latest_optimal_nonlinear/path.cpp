#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "Header.h"

using namespace cv;
using namespace std;

vector<Point> path;
vector<Point> RWall;
vector<Point> LWall;

vector<Point2f> fRWall;
vector<Point2f> fLWall;

const float r = 1.0 / 70.0;

vector<vector<Point>> paths;
bool drawing = false;


void DRAW_callback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDBLCLK) {

        drawing = !drawing;

        if (!drawing) {
            destroyAllWindows();
        }
    }

    if (event == EVENT_MOUSEMOVE && drawing) {
        path.push_back(Point(x, y));
    }
}


void show(void) {

	Mat img;

    img = Mat::zeros(Size(700, 700), CV_8U);
	imshow("Right Wall", img);
    setMouseCallback("Right Wall", DRAW_callback);
	waitKey(0);

    ofstream F("R.txt", ios_base::trunc);

    for (Point P : path) {
        F << P.x << ',' << P.y << endl;
    }

    F.clear();
    F.close();

    paths.push_back(path);
    polylines(img, paths, false, Scalar(255, 255, 255), 3);
    imwrite("R.png", img);

    RWall = path;
    path.clear();
    paths.clear();

    img = Mat::zeros(Size(700, 700), CV_8U);
    imshow("Left Wall", img);
    setMouseCallback("Left Wall", DRAW_callback);
    waitKey(0);

    F.open("L.txt", ios_base::trunc);

    for (Point P : path) {
        F << P.x << ',' << P.y << endl;
    }

    F.clear();
    F.close();

    paths.push_back(path);
    polylines(img, paths, false, Scalar(255, 255, 255), 3);
    imwrite("L.png", img);

    LWall = path;
    path.clear();
    paths.clear();

    for (Point P : RWall) {
        fRWall.push_back( Point2f((P.x - 350) * r, (350 - P.y) * r) );
    }

    for (Point P : LWall) {
        fLWall.push_back( Point2f((P.x - 350) * r, (350 - P.y) * r) );
    }

    RWall.clear();
    LWall.clear();

}

void load(void) {

    ifstream F;
    string line;
    string item;
    int x = 0;
    int y = 0;

    path.clear();
    F.open("R.txt", ios_base::in);

    while (getline(F, line)) {
        stringstream ss(line);

        if (getline(ss, item, ',')) {
            x = std::stoi(item);
        }

        
        if (getline(ss, item, ',')) {
            y = std::stoi(item);
        }

        path.push_back(Point(x, y));
    }

    RWall = path;

    for (Point P : RWall) {
        fRWall.push_back(Point2f((P.x - 350) * r, (350 - P.y) * r));
    }

    RWall.clear();
    F.clear();
    F.close();


    //---------------------------------------------------------------------

    path.clear();
    F.open("L.txt", ios_base::in);

    while (getline(F, line)) {
        stringstream ss(line);

        if (getline(ss, item, ',')) {
            x = std::stoi(item);
        }


        if (getline(ss, item, ',')) {
            y = std::stoi(item);
        }

        path.push_back(Point(x, y));
    }

    LWall = path;

    for (Point P : LWall) {
        fLWall.push_back(Point2f((P.x - 350) * r, (350 - P.y) * r));
    }

    LWall.clear();
    F.clear();
    F.close();
}

float dR(const float& x, const float& y) {

    static float d = 0.0;
    static float temp = 0.0;
    d = hypot(fRWall[0].x - x, fRWall[0].y - y);

    for (Point2f P : fRWall) {
        temp = hypot(P.x - x, P.y - y);

        if (temp < d) {
            d = temp;
        }
    }

    return d;
}


float dL(const float& x, const float& y) {

    static float d = 0.0;
    static float temp = 0.0;
    d = hypot(fLWall[0].x - x, fLWall[0].y - y);

    for (Point2f P : fLWall) {
        temp = hypot(P.x - x, P.y - y);

        if (temp < d) {
            d = temp;
        }
    }

    return d;
}