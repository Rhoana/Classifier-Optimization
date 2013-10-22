#include <unistd.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <fstream>
#include <getopt.h>
#include <assert.h>
#include <string>

using namespace cv;
using namespace std;

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <pthread.h> // for mutex

void adapthisteq(const Mat &in, Mat &out, float regularizer);
void local_statistics(Mat &image_in, int windowsize, Mat &mean, Mat &var, Mat deciles[10]);

void dog_2_50(const Mat &in, Mat &out)
{
    Mat blur_2, blur_50, diff(in.size(), CV_16S);
    cilk_spawn GaussianBlur(in, blur_2, Size(0, 0), 2);
    cilk_spawn GaussianBlur(in, blur_50, Size(0, 0), 50);
    cilk_sync;
    diff = blur_2 - blur_50;
    diff.convertTo(out, CV_8U, 0.5, 255);
}

static int verbose;

int main(int argc, char** argv) {
  int numWorkers = __cilkrts_get_nworkers();
  cout << "number of cilk workers = " << numWorkers << endl;

  int windowsize = 19;
  int blocksize = 8;
  verbose = 1;
  
  assert(argc == 4);

  string input_image(argv[1]);
  char *classifier_file = argv[2];
  char *output_image = argv[3];

  cout << "Classifying and storing features from " << input_image << " in " << output_image << endl;
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread(input_image, 0);
  image.convertTo(image, CV_8U);
  
  /* normalize image */
  Mat adapt_image(image.size(), CV_8U);
  adapthisteq(image, adapt_image, 2);  // max CDF derivative of 2

  /* FEATURE: local statistics: mean, variance, and pixel counts per-decile */
  Mat local_mean, local_var, deciles[10];
  cilk_spawn local_statistics(image, windowsize,
                              local_mean, local_var, deciles);

  /* FEATURE: DoG_2_50 */
  Mat DoG_2_50;
  cilk_spawn dog_2_50(adapt_image, DoG_2_50);

  /* FEATURE: Blur 10 */
  Mat blur_10;
  cilk_spawn GaussianBlur(adapt_image, blur_10, Size(0, 0), 10);

  /* Make prediction */
  Mat classifier_image;
  imread(classifier_file, 0).convertTo(classifier_image, CV_32F);

  /* wait for features to finish */
  cilk_sync;

  /* NB: Hard coded ordering by feature name as stored in python notebook */
  uchar *feature_ptrs[16];
  feature_ptrs[0] = adapt_image.ptr(0);
  feature_ptrs[1] = blur_10.ptr(0);
  for (int i = 0; i < 10; i++)
      feature_ptrs[2 + i] = deciles[i].ptr(0);
  feature_ptrs[12] = local_mean.ptr(0);
  feature_ptrs[13] = local_var.ptr(0);
  feature_ptrs[14] = image.ptr(0); /* Feature name: "original" */
  feature_ptrs[15] = DoG_2_50.ptr(0); /* capitalization */

  Mat prediction = Mat::zeros(image.size(), CV_32F);
  
  float *p = prediction.ptr<float>(0);
  for (int i = 0; i < prediction.total(); i++)
      for (int feature_idx = 0; feature_idx < 16; feature_idx++) { 
          p[i] += classifier_image.at<float>(feature_idx, feature_ptrs[feature_idx][i]);
      }

  /* adjust features from logistic to probability */
  prediction = -prediction;
  exp(prediction, prediction);
  prediction = 1.0 / (1.0 + prediction);
}
