// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FeaturePoints.h"

using namespace std;

string end_txt = ".txt";
string end_sift = ".sift";

void txt2sift();
void sift2txt();

int main()
{   
    txt2sift();



    /*FeatureData test;
    const char* sift_file= "C:/Users/gr050/Desktop/sift/2.sift";
    out = test.ReadSIFTB(sift_file);

    test.print_test();*/

    return 0;

}

void sift2txt() {
    int out;
    string saveTxtF = "C:\\Users\\gr050\\Desktop\\DATASET\\orig_sift2txt\\";
    string OrigSiftF = "C:\\Users\\gr050\\Desktop\\DATASET\\TUM_images\\";
    FeatureData test;
    for (int i = 0; i < 371; i++) {
        string name = to_string(i);
        string tmp_save_ = OrigSiftF + name + end_sift;
        const char* temp = tmp_save_.c_str();
        test.ReadSIFTB(temp);
        test.Sift2Txt(saveTxtF + name + end_txt);
    }
}


void txt2sift() {
    int out;
    string save_folder = "C:/Users/gr050/Desktop/DATASET/sift_/";
    string read_folder = "C:/Users/gr050/Desktop/DATASET/sift/";

    bool loc_des = true;
    FeatureData test;
    for (int i = 0; i < 1000; i++) {
        string name = to_string(i);
        string tmp_save_ = save_folder + name + end_sift;
        const char* tmp_save = tmp_save_.c_str();
        test.ReadLocFromText(read_folder + name + end_txt);
        test.saveSIFTB2(tmp_save);
    }
}