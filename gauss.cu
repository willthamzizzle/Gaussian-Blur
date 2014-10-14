#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cutil.h>
#include <cutil_inline.h>

// to display in color
#define RED "\33[1;31m"
#define GREEN "\33[22;32m"
#define BLUE "\33[1;34m"
#define WHITE "\33[1;37m"
#define NC "\33[0m"		// no color

#define e 2.71828183
#define PI 3.1415

#define	START_TIME (startTime.tv_sec * 1000000 + startTime.tv_usec)/1000000.0	// time in seconds
#define END_TIME   (endTime.tv_sec   * 1000000 + endTime.tv_usec)/1000000.0		// time in seconds

typedef struct {
	short int blue;		// order is BGR in bmp format
	short int green;
	short int red;
	short int alpha;
} Pixel;

struct _fileInfo{
	double size;	// size of file
	int offset;		// offset to bitmap data
	int width;		// width of bitmap in pixels
	int height;		// height of bitmap in pixels
	int bitDepth;	// number of bits per pixel (ex: 24bit, 32bit)
	int data;		// number of pixels
	int pixels;		// number of pixels in image
} FileInfo;

struct timeval startTime, endTime;


/**
 * Initializes bitmap file header data
 */
void buildHeader(char *bmpHeader) {
	FileInfo.size = *((long *)(bmpHeader + 2));
	FileInfo.offset = *((long *)(bmpHeader + 10));
	FileInfo.width = *((long *)(bmpHeader + 18));
	FileInfo.height = *((long *)(bmpHeader + 22));
	FileInfo.bitDepth = *((short *)(bmpHeader + 28));
	//FileInfo.data = *((long *)(bmpHeader + 34));

	if (FileInfo.height < 0) {	// change value to positive
		FileInfo.height *= -1;	// negative means data is top to bottom instead of bottom to top
	}
	if (FileInfo.bitDepth == 32) {	// 32 bit image
		FileInfo.data = 4*FileInfo.width * FileInfo.height;
	} else {						// 24 bit image
		FileInfo.data = 3*(FileInfo.width * FileInfo.height) + (4-((FileInfo.width*3)%4))*FileInfo.height;
	}															// add padding for each row
}

/**
 * Build color array
 */
void initColors(char *pixelBuffer, Pixel *srcPixel) {
	int i,j,k=0, l=0;


	if (FileInfo.bitDepth == 32) {	// 32 bit image
		for (i=0; i<FileInfo.data; i+=4) {
			srcPixel[k].blue  = *(pixelBuffer + i) & 0xFF;
			srcPixel[k].green = *(pixelBuffer + i+1) & 0xFF;
			srcPixel[k].red   = *(pixelBuffer + i+2) & 0xFF;
			srcPixel[k].alpha = *(pixelBuffer + i+3) & 0xFF;
			k++;
		}
	} else {						// 24 bit image
		for (i=0; i<FileInfo.height; i++) {
			for (j=0; j<FileInfo.width; j++) {
				srcPixel[k].blue  = *(pixelBuffer + l++) & 0xFF;
				srcPixel[k].green = *(pixelBuffer + l++) & 0xFF;
				srcPixel[k].red   = *(pixelBuffer + l++) & 0xFF;
				k++;
			}
			if ((FileInfo.width*3)%4 != 0) {
				l+=4-((FileInfo.width*3)%4);	// add padding at end of each row
			}
		}
	}
	FileInfo.pixels = k;	// record total number of pixels in image.
}

/**
 * Creates a bitmap image file with then new data
 */
void writeBMP(FILE *outputFilePtr, char *bmpHeader, Pixel *dstPixel) {
	int i,j;

	// write header info
	fwrite(bmpHeader,1,54,outputFilePtr);

	// write color data to file
	if (FileInfo.bitDepth == 32) {	// 32 bit image
		for (i=0; i<FileInfo.width * FileInfo.height; i++) {
			fputc(dstPixel[i].blue,  outputFilePtr);
			fputc(dstPixel[i].green, outputFilePtr);
			fputc(dstPixel[i].red,   outputFilePtr);
			fputc(dstPixel[i].alpha, outputFilePtr);
		}
	} else {						// 24 bit image
		for (i=0; i<FileInfo.width * FileInfo.height; i++) {
			fputc(dstPixel[i].blue,  outputFilePtr);
			fputc(dstPixel[i].green, outputFilePtr);
			fputc(dstPixel[i].red,   outputFilePtr);

			// add padding at end of each row
			if (i > 0 && i % FileInfo.width == 0 && (FileInfo.width*3)%4 != 0) {
				for (j=0; j<4-((FileInfo.width*3)%4); j++) {
					fputc(0x0, outputFilePtr);
				}
			}
		}
	}
	// write additional zeros at end if file does not contain a multiple of 4 bytes
	if (FileInfo.width%4 != 0) {
		for (i=0; i<FileInfo.width%4; i++) {
			fputc(0x0, outputFilePtr);
		}
	}
}

/**
 * Displays bitmap image information
 */
void displayFileInfo()
{
	printf("_________________________________\n");
	printf("  FILE INFORMATION\n");
	printf("---------------------------------\n");
	if ((FileInfo.size / 1024) >= 1000) {
		printf("  File size:     %.2f MB\n",FileInfo.size / 1048576);
	} else {
		printf("  File size:     %.2f KB\n",FileInfo.size / 1024);
	}
	printf("  Data Offset:   %i bytes\n",FileInfo.offset);
	printf("  Image Width:   %i px\n",FileInfo.width);
	printf("  Image Height:  %i px\n",FileInfo.height);
	printf("  Bit Depth:     %i bit\n",FileInfo.bitDepth);
	printf("  BMP Data:      %i bytes\n",FileInfo.data);
	printf("---------------------------------\n");
}

/**
 * Display bitmap color data
 */
void displayColorValues(Pixel *srcPixel) {
	int i,j,k=0;
	int dst;

	if (FileInfo.bitDepth == 32) {	// 32 bit image
		printf(" PIXEL #  RED  GREEN BLUE  ALPHA\n");
		for (i=0; i<FileInfo.height; i++) {
			for (j=0; j<FileInfo.width; j++) {
				dst = FileInfo.width*i+j;
				printf("%6i: "RED"%5i "GREEN"%5i "BLUE"%5i "WHITE"%5i"NC"\n", k,
					   srcPixel[dst].red,
					   srcPixel[dst].green,
					   srcPixel[dst].blue,
					   srcPixel[dst].alpha);
				k++;
			}
		}
	} else {						// 24 bit image
		printf(" PIXEL #  RED  GREEN BLUE\n");
		for (i=0; i<FileInfo.height; i++) {
			for (j=0; j<FileInfo.width; j++) {
				dst = FileInfo.width*i+j;
				printf("%6i: "RED"%5i "GREEN"%5i "BLUE"%5i"NC"\n", k,
					   srcPixel[dst].red,
					   srcPixel[dst].green,
					   srcPixel[dst].blue);
				k++;
			}
		}
	}
}
//============================================================================
//-------------------CUDA CODE------------------------------------------------
__global__ void convolutionRowsKernel(short int* d_cudaSrc, short int* d_cudaDst)//, float* cudaKernel, int sigma, int width, int height)
{
	int i, j, k, dstIndex;

	i = (blockDim.x * blockIdx.x) + threadIdx.x;
	j = (blockDim.x * blockIdx.y) + threadIdx.y;
	dstIndex = (i * 8192) + j;

	d_cudaDst[dstIndex] = 112;
}


__global__ void convolutionColumnsKernel(short int* d_cudaSrc, short int* d_cudaDst)//, float* cudaKernel, int sigma, int width, int height)
{
	d_cudaDst[1] = 101;
}

//============================================================================
//-------------------END CUDA CODE------------------------------------------------


float *makeKernel1D(int sigma)
{
	int i;
	double r;	// radius
	float sigma2 = sigma * sigma;
	float sum;

	float *kernel;
	kernel = (float *)malloc((2 * sigma + 1)*sizeof(float));

	for(i = 0; i < (2*sigma+1); i++)
	{
		r = sigma - i;
		kernel[i] = (float) exp(-0.5 * (r*r) / sigma2);
		sum += kernel[i];
	}

	for (i=0; i<2*sigma+1; i++) {
		kernel[i] /= sum;
		//printf("%f ",kernel[i]);
	}
	return kernel;
}


/**
 * Returns the differnce of the startTime and endTime in milliseconds or seconds
 */
float elapsedTime() {
	return END_TIME - START_TIME;
}


// ===================================================================================================
int main(int argc, char* argv[])
{
	int i, j, k;

	//*****FILE in/out Variables*****
	FILE *inputFilePtr, *outputFilePtr;		// file pointers
	char *bmpHeader;						// 54 byte header buffer
	char *pixelBuffer;						// pixel color source buffer
	size_t result;							// total number of bytes read by fread()
	Pixel *srcPixel;						// pixel source
	Pixel *dstPixel;
	//*******************************

	int width;
	int height;
	int sigma;							// pixel distance to average from point in gassian formula
	float *kernel1D;						// our matrix to convolve with



	float seqT =  0.0;						// sequential timing result
	float ompT =  0.0;						// openMP timing result
	float cudaT = 0.0;						// CUDA timing result

	float temp = 32;


	/*************************************************************************************************
	**********CUDA VARIABLES ONLY!!!!*****************************************************************
	*************************************************************************************************/
	short int *d_cudaSrcR;
	short int *d_cudaSrcG;
	short int *d_cudaSrcB;

	short int *d_cudaDstR;
	short int *d_cudaDstG;
	short int *d_cudaDstB;

	float *d_cudaKernel;

	short int *h_cudaSrcR;
	short int *h_cudaSrcG;
	short int *h_cudaSrcB;

	short int *h_cudaDstR;
	short int *h_cudaDstG;
	short int *h_cudaDstB;

	int THREAD_SIZE = 256;
	int BLOCK_SIZE = 32;
	/*************************************************************************************************
	/*************************************************************************************************/

	// -----------------------------------------------------------------------------------------------
	bmpHeader = (char*)malloc(54 * sizeof(char));	// allocate bmp header space

	// check for valid input
	if (argc != 2) {
		printf("Invalid Input:  Missing file name!\n");
		exit(0);
	}

	// open input and output files
	inputFilePtr = fopen(argv[1], "rb");		// open binary file for reading
	outputFilePtr = fopen("cudaOutputImage.bmp", "wb");	// open binary file for writing

	if (!inputFilePtr) {
		printf("Unable to open file %s!\n", argv[1]);
		exit(0);
	}

	// read bmp header info
	if ((result = fread(bmpHeader,1,54,inputFilePtr)) != 54) {
		printf("File Error:  Invalid bitmap header.\n");
		exit(0);
	}

	// check to see if the file is a bmp image
	if (bmpHeader[0] != 'B' && bmpHeader[1] != 'M') {
		printf("File Error:  Invalid bitmap file.\n");
		exit(0);
	}

	buildHeader(bmpHeader);	// get header info

	// rgb data must begin at byte 54
	if (FileInfo.offset != 54) {
		printf("File does not contain a valid bitmap header.\n");
		exit(0);
	}

	// allocate memory
	pixelBuffer = (char *)malloc(FileInfo.data);	// enough space for all rgb values
	srcPixel = (Pixel *)malloc(FileInfo.width * FileInfo.height * sizeof(Pixel));

	// read color data ifrom file
	result = fread(pixelBuffer,1,FileInfo.data, inputFilePtr);
	fclose(inputFilePtr);	// close input file

	displayFileInfo();								// display image information
	initColors(pixelBuffer, srcPixel);				// initialize srcPixel with rgb values
	//displayColorValues(srcPixel);					// display pixel color values


	// get sigma value
	printf("Enter the blur radius: ");
	if (scanf("%i",&sigma) != EOF) {
		if (sigma > 0 && sigma < (FileInfo.width/2) && sigma < (FileInfo.height/2)) {

		}
		else {
			sigma = 1;
		}
	}

	kernel1D = makeKernel1D(sigma);


	//========================CUDA CODE=========================================
	//--------------------------------------------------------------------------

	width = FileInfo.width;
	height = FileInfo.height;

	h_cudaSrcR = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);
	h_cudaSrcG = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);
	h_cudaSrcB = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);

	h_cudaDstR = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);
	h_cudaDstG = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);
	h_cudaDstB = (short int *)malloc(sizeof(short int *) *FileInfo.width * FileInfo.height);

	for(int i = 0; i < width*height; i++)
	{
		h_cudaSrcR[i] = srcPixel[i].red;
		h_cudaSrcG[i] = srcPixel[i].green;
		h_cudaSrcB[i] = srcPixel[i].blue;

		h_cudaDstR[i] = 0;
		h_cudaDstG[i] = 0;
		h_cudaDstB[i] = 0;
	}

	cudaMalloc((void**)&d_cudaSrcR, (FileInfo.width * FileInfo.height) * sizeof(short int));
	cudaMalloc((void**)&d_cudaSrcG, (FileInfo.width * FileInfo.height) * sizeof(short int));
	cudaMalloc((void**)&d_cudaSrcB, (FileInfo.width * FileInfo.height) * sizeof(short int));

	cudaMalloc((void**)&d_cudaDstR, (FileInfo.width * FileInfo.height) * sizeof(short int));
	cudaMalloc((void**)&d_cudaDstG, (FileInfo.width * FileInfo.height) * sizeof(short int));
	cudaMalloc((void**)&d_cudaDstB, (FileInfo.width * FileInfo.height) * sizeof(short int));

	cudaMalloc((void**)&d_cudaKernel, ((sigma*2+1))*sizeof(float));

	cudaMemcpy(d_cudaSrcR, h_cudaSrcR, (FileInfo.width * FileInfo.height) * sizeof(short int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cudaSrcG, h_cudaSrcG, (FileInfo.width * FileInfo.height) * sizeof(short int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cudaSrcB, h_cudaSrcB, (FileInfo.width * FileInfo.height) * sizeof(short int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_cudaKernel, kernel1D, (sigma*2+1) * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blocks(256, 256, 1); 	//A lot of blocks
  	dim3 threads(16, 16, 1);	//256 threads

  	cudaEvent_t start_event, stop_event;
  //	CUDA_SAFE_CALL( cudaEventCreate(&start_event) );	*/
  	CUDA_SAFE_CALL( cudaEventCreate(&stop_event) ); /*
  	cudaEventRecord(start_event, 0);*/

  	convolutionRowsKernel<<<blocks, threads>>>(d_cudaSrcR, d_cudaDstR);//, cudaKernel, sigma, FileInfo.width, FileInfo.height);
  	//convolutionColumnsKernel<<<blocks, threads>>>(cudaSrcPixelR, cudaDstPixelR);//, cudaKernel, sigma, FileInfo.width, FileInfo.height);

  	//cudaEventRecord(stop_event, 0);
 	cudaEventSynchronize(stop_event);
  //	CUDA_SAFE_CALL( cudaEventElapsedTime(&cudaT,start_event, stop_event) );

  	//Copy back from device memory to main memory
	cudaMemcpy(h_cudaDstR, d_cudaDstR, (width * height * sizeof(short int)), cudaMemcpyDeviceToHost);
  	//========================END CUDA CODE======================================

	for(int i = 0; i < 4; i++)
	{

		printf("\t Red: %d ", h_cudaDstR[i]);

	}

	printf("\n**********************************************************\n");

	for(int i = 0; i < 4; i++)
	{
		printf("\t Red: %d ", srcPixel[i].red);
	}


	// display timing results
	printf("\n\nSequential Time:  %.4f seconds\n",seqT);
	printf("Open MP Time:     %.4f seconds (%.2f x sequential)\n",ompT,seqT/ompT);
	printf("CUDA Time:        %.4f seconds (%.2f x sequential)\n",cudaT,seqT/cudaT);
	printf("\n");

	// write new bitmap image to disk
	//writeBMP(outputFilePtr, bmpHeader, out_cudaDstPixel);

	fclose(outputFilePtr);	// close output file
}
