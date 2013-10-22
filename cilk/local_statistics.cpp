#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace cv;
using namespace std;

void local_statistics(Mat &image_in, int windowsize, Mat &mean, Mat &var, Mat deciles[10])
{
    Mat image;
    image_in.convertTo(image, CV_32F);

    // we compute statistics using a Guassian blur, rather than a hard window
    GaussianBlur(image, mean, Size(0, 0), windowsize);
    GaussianBlur(image.mul(image), var, Size(0, 0), windowsize);
    var = var - mean.mul(mean);
    mean.convertTo(mean, CV_8U);
    var.convertTo(var, CV_8U);
    
    // histogram features
    cilk_for (int i = 0; i < 10; i++) {
        float threshold_lo = i * 256.0 / 10.0;
        float threshold_hi = (i+1) * 256.0 / 10.0;
        Mat mask = (image >= threshold_lo).mul(image < threshold_hi);
        Mat tmp;
        mask.convertTo(mask, CV_32F);
        GaussianBlur(mask, tmp, Size(0, 0), windowsize);
        tmp.convertTo(deciles[i], CV_8U);
    }
}
